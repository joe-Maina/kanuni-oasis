"""Run the Kanuni AgentSociety 2 simulation: the general public's offline life.

Each step (one day, or a week in quiet periods) runs:
  1. modules get the step's date (news, prices, protests for that date);
  2. each agent lives the step: AS2's PersonAgent observes (our observation:
     news via its channels, its household money, protests today) and acts
     through the env router (talk to people, change spending or work,
     borrow, protest, contact the MP);
  3. each agent answers our strict stance survey (not stored in memory);
  4. the modules book the step: income, costs, missed payments, messages.

We run our own step loop instead of AgentSociety.step(), which raises if any
agent fails: here a failure is recorded for that agent and the rest carry on.
The router runs in-process (no Ray), so our env modules are plain imports.
AS2 core code is not modified.

Usage (from agentic-society/, with the Ollama shim running):
    .venv/bin/python -m kanuni_as2.run --config kanuni_as2/configs/tiny_5.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROFILE_KEYS = ["name", "age", "gender", "sub_county", "county",
                "urban_rural", "ethnicity", "role", "livelihood", "income",
                "education", "trust_in_government", "media_habits"]

WORLD_TEXT = (
    "You are an ordinary person living in Kenya. Each day you go about your "
    "life: work, family, money, church or mosque, and the news. You hear "
    "about national politics through the radio, TV, WhatsApp, social media "
    "and people around you. You have a household budget with real limits: "
    "you cannot spend money you do not have, and rent, school fees and "
    "loans fall due on fixed dates.\n"
    "To do things in the world, use ask_env with a clear instruction that "
    "includes your agent id. You can: tell neighbours, workmates or family "
    "what you think; post in your WhatsApp group; post on social media you "
    "use; change your household spending plan (normal, cut_back, "
    "cheaper_goods); change today's work plan (normal, work_extra, "
    "take_day_off); borrow money (mobile_loan, chama, family, shylock); pay "
    "an unpaid bill or loan; join a protest; contact your MP; or speak at a "
    "community meeting. You may also do nothing beyond your usual routine. "
    "Your observation already tells you what you heard today and your money "
    "situation; you do not need to ask the environment for them. Do what "
    "this person would do today, then finish the day.\n"
    "Act as the person you are, not as an assistant.")


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    for key in ("personas", "profiles", "channels", "scenario"):
        cfg[key] = str((path.parent / cfg[key]).resolve())
    return cfg


def set_llm_env(cfg: dict) -> None:
    """AS2 reads its model settings from env vars at import time."""
    m = cfg["model"]
    os.environ["AGENTSOCIETY_LLM_API_KEY"] = "unused"
    os.environ["AGENTSOCIETY_LLM_API_BASE"] = m["url"]
    os.environ["AGENTSOCIETY_LLM_MODEL"] = m["name"]
    os.environ["AGENTSOCIETY_LLM_REQUEST_TIMEOUT"] = str(
        m.get("request_timeout_s", 600))
    os.environ.setdefault("MEM0_TELEMETRY", "False")
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


def step_dates(cfg: dict) -> list[tuple[date, int]]:
    """(date, days covered) for every step, from cfg['schedule']."""
    out = []
    for block in cfg["schedule"]:
        d = date.fromisoformat(block["from"])
        end = date.fromisoformat(block["to"])
        while d <= end:
            n = min(block["step_days"], (end - d).days + 1)
            out.append((d, n))
            d += timedelta(days=n)
    return out


class TransportRetryCounter(logging.Handler):
    """Counts AS2's own transport retries (logged warnings), so they are reported."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.count = 0

    def emit(self, record):
        msg = record.getMessage()
        if msg.startswith(("Request failed for", "Rate limit for")):
            self.count += 1


INFRA_MARKERS = ("Timeout", "timed out", "Connection", "APIConnectionError",
                 "ServiceUnavailable", "InternalServerError", "502", "503")


def classify_exception(e: Exception) -> str:
    text = f"{type(e).__name__}: {e}"
    cause = e.__cause__
    if cause is not None:
        text += f" | {type(cause).__name__}: {cause}"
    if "context_overflow" in text:
        # The shim refused a prompt too long for the context window.
        return "context_overflow"
    return "infra_error" if any(m in text for m in INFRA_MARKERS) \
        else "llm_error"


async def run(cfg: dict) -> Path:
    run_dir = HERE / "runs" / cfg["run_id"]
    if run_dir.exists():
        raise SystemExit(f"{run_dir} exists; choose a new run_id.")
    run_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(json.dumps(cfg, indent=2))
    set_llm_env(cfg)
    os.environ["AGENTSOCIETY_HOME_DIR"] = str(run_dir / "as2_home")

    # Imported after the env vars are set.
    from agentsociety2.agent.service_proxy import build_service_proxy
    from agentsociety2.env.router_react import ReActRouter
    from agentsociety2.logger import get_logger

    from kanuni_as2.agent import KanuniPersonAgent
    from kanuni_as2.common import World, append_jsonl, load_json
    from kanuni_as2.envs.civic import KanuniCivicEnv
    from kanuni_as2.envs.household import KanuniHouseholdEnv
    from kanuni_as2.envs.information import KanuniInformationEnv
    from kanuni_as2.stance import interview_prompt, parse_interview

    retries = TransportRetryCounter()
    get_logger().addHandler(retries)
    log_file = logging.FileHandler(run_dir / "as2.log")
    log_file.setLevel(logging.INFO)
    get_logger().addHandler(log_file)

    personas = load_json(cfg["personas"])
    if cfg.get("connectivity_filter"):
        personas = [p for p in personas
                    if p["connectivity"] in cfg["connectivity_filter"]]
    if cfg.get("agent_ids"):
        keep = set(cfg["agent_ids"])
        personas = [p for p in personas if p["agent_id"] in keep]
    if cfg.get("max_agents"):
        personas = personas[:cfg["max_agents"]]
    ids = [p["agent_id"] for p in personas]

    world = World(cfg["personas"], cfg["profiles"], cfg["channels"],
                  cfg["scenario"], cfg["seed"], agent_ids=ids)
    log_path = str(run_dir / "env_log.jsonl")
    info = KanuniInformationEnv(world, log_path)
    household = KanuniHouseholdEnv(world, log_path)
    civic = KanuniCivicEnv(world, log_path)
    household.civic = civic
    rcfg = cfg.get("router", {})
    router = ReActRouter(env_modules=[info, household, civic],
                         max_steps=rcfg.get("max_steps", 4),
                         max_llm_call_retry=1)
    # A fixed, reviewed world description instead of one the model writes.
    router._world_description = WORLD_TEXT

    steps = step_dates(cfg)
    start = datetime.combine(steps[0][0], datetime.min.time())
    await router.init(start)
    proxy = build_service_proxy(router, run_dir=run_dir, trace=False,
                                replay=False)

    ws_root = run_dir / "agents"
    for p in personas:
        profile = {"id": p["agent_id"],
                   **{k: p[k] for k in PROFILE_KEYS if k in p}}
        profile["media_habits"] = ", ".join(p["media_habits"])
        KanuniPersonAgent.create(ws_root / f"agent_{p['agent_id']:04d}",
                                 profile, cfg.get("agent_config", {}))
    (run_dir / "channels.json").write_text(json.dumps(
        {"channels": world.channels, "ties": world.ties}, indent=2))

    prompt = interview_prompt(cfg["bill_name"])
    sem = asyncio.Semaphore(cfg["max_concurrency"])
    records = run_dir / "agent_steps.jsonl"

    async def agent_step(aid: int, step_idx: int, day: date, tick: int,
                         t: datetime):
        ws = ws_root / f"agent_{aid:04d}"
        rec = {"step_index": step_idx, "date": day.isoformat(),
               "days": tick // 86400, "agent_id": aid}
        async with sem:
            t0 = time.time()
            agent = await KanuniPersonAgent.from_workspace(ws, proxy)
            agent.reset_counters()
            try:
                summary = await agent.step(tick, t)
                status = "no_valid_finish" if agent.kanuni_counts[
                    "fallback"] else "ok"
                error = None
            except Exception as e:  # counted, not retried
                summary, status = None, classify_exception(e)
                error = f"{type(e).__name__}: {e}"[:500]
            rec.update(action_status=status, action_error=error,
                       step_summary=summary,
                       step_counts=dict(agent.kanuni_counts),
                       step_model_errors=agent.kanuni_errors[:5],
                       action_s=round(time.time() - t0, 1))
            agent.reset_counters()
            t1 = time.time()
            try:
                raw = await agent.ask(prompt, readonly=True, t=t)
                if agent.kanuni_counts["fallback"]:
                    # AS2 made up an answer because the model never
                    # finished; not the model's own reply.
                    parsed = {"ok": False, "stance": None, "intent": None,
                              "reason": None,
                              "error": "no finish; AS2 fallback answer"}
                    iv_status = "no_valid_finish"
                else:
                    parsed = parse_interview(raw)
                    iv_status = "ok" if parsed["ok"] else "parse_failed"
                iv_error = parsed["error"]
            except Exception as e:
                raw = None
                parsed = {"ok": False, "stance": None, "intent": None,
                          "reason": None}
                iv_status = classify_exception(e)
                iv_error = f"{type(e).__name__}: {e}"[:500]
            rec.update(interview_status=iv_status, interview_raw=raw,
                       interview_error=iv_error, stance=parsed["stance"],
                       intent=parsed["intent"], reason=parsed["reason"],
                       interview_counts=dict(agent.kanuni_counts),
                       interview_s=round(time.time() - t1, 1))
            try:
                await agent.to_workspace(ws)
                await agent.close()
            except Exception as e:
                rec["persist_error"] = f"{type(e).__name__}: {e}"[:300]
        append_jsonl(records, rec)
        return rec

    for step_idx, (day, n_days) in enumerate(steps):
        t = datetime.combine(day, datetime.min.time())
        tick = n_days * 86400
        router.set_current_time(t)
        before = retries.count
        t_step = time.time()
        recs = await asyncio.gather(*[
            agent_step(aid, step_idx, day, tick, t) for aid in ids])
        await router.step(tick, t)
        n_ok = sum(r["action_status"] == "ok" for r in recs)
        n_iv = sum(r["interview_status"] == "ok" for r in recs)
        print(f"[{day}] step {step_idx} ({n_days}d): {n_ok}/{len(recs)} "
              f"steps finished, {n_iv}/{len(recs)} parsed interviews, "
              f"{retries.count - before} transport retries, "
              f"{time.time() - t_step:.0f}s", flush=True)
        append_jsonl(run_dir / "steps.jsonl", {
            "step_index": step_idx, "date": day.isoformat(),
            "days": n_days, "transport_retries": retries.count - before,
            "seconds": round(time.time() - t_step, 1)})

    await router.close()
    return run_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run-id", help="override the config's run_id")
    a = p.parse_args()
    cfg = load_config(Path(a.config).resolve())
    if a.run_id:
        cfg["run_id"] = a.run_id
    run_dir = asyncio.run(run(cfg))
    from kanuni_as2.export import export_run
    print(json.dumps(export_run(run_dir), indent=2))


if __name__ == "__main__":
    main()
