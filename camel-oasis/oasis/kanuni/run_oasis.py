"""Run the Kanuni OASIS simulation: personas react to dated Finance Bill events.

One tick = one simulated day. Each day:
  1. media accounts post the events dated that day (on day 0, also every
     event dated before the start date);
  2. the recommendation table is rebuilt;
  3. each persona agent observes its feed and acts once via the LLM;
  4. each persona agent is interviewed for stance and intent (not stored in
     its memory, so the survey does not steer later behaviour).

We drive the day loop ourselves instead of calling OasisEnv.step(), because
step() discards each agent's result and we need exact per-agent failure
counts. Platform setup and sign-up still go through OasisEnv. OASIS core code
is not modified.

Usage (from the repo root, with kanuni.ollama_shim running):
    python -m kanuni.run_oasis --config kanuni/configs/smoke_100.json
"""
import argparse
import asyncio
import json
import os
import random
import time
import warnings
from datetime import date, timedelta
from pathlib import Path

from camel.models import ModelFactory
from camel.prompts import TextPrompt
from camel.types import ModelPlatformType

from kanuni.stance import INTERVIEW_PROMPT, parse_interview

REPO = Path(__file__).resolve().parent.parent
# CAMEL warns on every call once the memory window is full; that is intended.
warnings.filterwarnings("ignore", message="Chat history window size limit")

PERSONA_TEMPLATE = TextPrompt(
    "# OBJECTIVE\n"
    "You are a Kenyan user of X (Twitter). Each day you see posts in your "
    "feed. Choose actions with the provided functions that fit who you are. "
    "You may write in English, Swahili or Sheng, as you normally would.\n\n"
    "# SELF-DESCRIPTION\n"
    "Your name is {name}. You are a {age}-year-old {gender} living in "
    "{sub_county}, {county} County ({urban_rural} area). Ethnicity: "
    "{ethnicity}. Occupation: {role}. Income: {income}. Education: "
    "{education}. Your trust in the national government is "
    "{trust_in_government}. You mainly use: {media_habits}.\n\n"
    "# RESPONSE METHOD\n"
    "Please perform actions by tool calling.")
PROFILE_KEYS = ["name", "age", "gender", "sub_county", "county",
                "urban_rural", "ethnicity", "role", "income", "education",
                "trust_in_government", "media_habits"]


def make_feed_env(action, max_comments: int):
    """OASIS's feed, with each post's comments cut to the most-liked few.

    OASIS shows every comment on every post, so with many agents a feed
    grows by thousands of tokens a day. X also shows only a few top replies.
    The platform still records the full feed in its trace.
    """
    import json as _json

    from oasis.social_agent.agent_environment import SocialEnvironment

    class CappedFeedEnvironment(SocialEnvironment):
        async def get_posts_env(self) -> str:
            posts = await self.action.refresh()
            if not posts["success"]:
                return "After refreshing, there are no existing posts."
            for post in posts["posts"]:
                comments = post.get("comments") or []
                if len(comments) > max_comments:
                    comments = sorted(
                        comments, key=lambda c: (c.get("num_likes", 0),
                                                 c.get("comment_id", 0)),
                        reverse=True)[:max_comments]
                    post["comments"] = comments
            return self.posts_env_template.substitute(
                posts=_json.dumps(posts["posts"], indent=4))

    return CappedFeedEnvironment(action)


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    for key in ("personas", "events"):
        cfg[key] = str((path.parent / cfg[key]).resolve())
    cfg["extra_timelines"] = [str((path.parent / x).resolve())
                              for x in cfg.get("extra_timelines", [])]
    return cfg


def make_model(cfg: dict):
    m = cfg["model"]
    return ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type=m["name"], url=m["url"], api_key="unused",
        model_config_dict={"temperature": m.get("temperature", 0.7)},
        timeout=m.get("timeout_s", 600),
        # No silent retries: failures are counted, not hidden.
        max_retries=0)


def write_jsonl(path: Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


# Transport failures (network, timeouts, shim/upstream errors) are not the
# model's fault; they are reported separately from model failures.
INFRA_ERRORS = ("APIConnectionError", "APITimeoutError", "InternalServerError",
                "APIStatusError")


def is_infra_error(exc: Exception) -> bool:
    return any(c.__name__ in INFRA_ERRORS for c in type(exc).__mro__)


def error_status(exc: Exception) -> str:
    """Status for an exception from a model call."""
    if "context_overflow" in str(exc):
        # The shim refused a prompt too long for the context window.
        return "context_overflow"
    return "infra_error" if is_infra_error(exc) else "llm_error"


def classify_llm_result(result) -> tuple[str, list, str | None]:
    """Map SocialAgent.perform_action_by_llm()'s return value to a status.

    It returns the CAMEL response if the model made a tool call, None if the
    model replied with text only, or the exception it caught.
    """
    if isinstance(result, Exception):
        return error_status(result), [], f"{type(result).__name__}: {result}"
    if result is None:
        return "no_tool_call", [], None
    calls = [{"name": t.tool_name, "args": t.args, "result": t.result}
             for t in result.info.get("tool_calls", [])]
    rejected = [c for c in calls if isinstance(c["result"], dict)
                and c["result"].get("success") is False]
    if calls and len(rejected) == len(calls):
        return "action_rejected", calls, None
    return "ok", calls, None


async def run(cfg: dict) -> Path:
    run_dir = REPO / "kanuni" / "runs" / cfg["run_id"]
    if run_dir.exists():
        raise SystemExit(f"{run_dir} exists; choose a new run_id.")
    run_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(json.dumps(cfg, indent=2))
    db_path = run_dir / "oasis.db"
    os.environ["OASIS_DB_PATH"] = str(db_path)
    os.chdir(run_dir)  # OASIS writes ./log relative to the cwd

    # Imported after chdir so OASIS's log files land in the run folder.
    import oasis
    from oasis import AgentGraph, SocialAgent, UserInfo
    from oasis.social_platform.channel import Channel

    random.seed(cfg["seed"])
    personas = [p for p in json.loads(Path(cfg["personas"]).read_text())
                if p["connectivity"] in cfg["connectivity_filter"]]
    if cfg.get("max_agents"):
        personas = personas[:cfg["max_agents"]]
    timeline = json.loads(Path(cfg["events"]).read_text())
    # Optional layers (e.g. activist and influencer posts) join the timeline.
    for extra_path in cfg["extra_timelines"]:
        extra = json.loads(Path(extra_path).read_text())
        timeline["media_accounts"] += extra["media_accounts"]
        timeline["events"] += extra["events"]
    model = make_model(cfg)
    actions = [oasis.ActionType(a) for a in cfg["available_actions"]]

    # OASIS needs user ids 0..N-1 with no gaps, so persona agent_ids are
    # mapped to OASIS ids; media accounts follow the personas.
    graph = AgentGraph()
    id_map = {}
    for oid, p in enumerate(personas):
        profile = {k: p[k] for k in PROFILE_KEYS}
        profile["media_habits"] = ", ".join(p["media_habits"])
        graph.add_agent(SocialAgent(
            agent_id=oid,
            user_info=UserInfo(user_name=p["user_name"], name=p["name"],
                               description=p["role"], profile=profile,
                               recsys_type="twitter"),
            user_info_template=PERSONA_TEMPLATE, agent_graph=graph,
            model=model, available_actions=actions))
        agent = graph.get_agent(oid)
        # Bounded prompts: capped comments per post, and memory limited to
        # the persona plus the most recent messages (CAMEL keeps the system
        # message). CAMEL cannot trim by tokens for this model without
        # summarising through extra model calls, so a message window is used.
        agent.env = make_feed_env(agent.env.action,
                                  cfg["max_comments_per_post"])
        agent.memory._window_size = cfg["memory_window_messages"]
        id_map[oid] = {"kind": "persona", "agent_id": p["agent_id"]}
    media_ids = {}
    for acct in timeline["media_accounts"]:
        oid = len(id_map)
        graph.add_agent(SocialAgent(
            agent_id=oid,
            user_info=UserInfo(user_name=acct["account_id"],
                               name=acct["name"], description=acct["bio"],
                               profile=None, recsys_type="twitter"),
            agent_graph=graph, model=model, available_actions=[]))
        id_map[oid] = {"kind": "media", "account_id": acct["account_id"],
                       "account_kind": acct.get("kind", "news")}
        media_ids[acct["account_id"]] = oid
    (run_dir / "id_map.json").write_text(json.dumps(id_map, indent=2))

    pcfg = cfg["platform"]
    platform = oasis.Platform(
        db_path=str(db_path), channel=Channel(),
        recsys_type=pcfg["recsys_type"],
        refresh_rec_post_count=pcfg["refresh_rec_post_count"],
        max_rec_post_len=pcfg["max_rec_post_len"],
        following_post_count=pcfg["following_post_count"])
    env = oasis.make(agent_graph=graph, platform=platform,
                     database_path=str(db_path))

    # Day -1 is setup: sign-up and following the media accounts. Exporters
    # ignore it. Calls go through the action layer, not agent memory.
    clock = platform.sandbox_clock
    clock.time_step = -1
    await env.reset()
    persona_agents = [(oid, graph.get_agent(oid)) for oid in id_map
                      if id_map[oid]["kind"] == "persona"]
    # Everyone follows the news accounts; other kinds of account (activists,
    # influencers, pro-government) are followed with the configured
    # probability, drawn per persona from the run seed.
    follow_p = cfg.get("follow_probability", {})
    for oid, agent in persona_agents:
        for mid in media_ids.values():
            kind = id_map[mid]["account_kind"]
            p_follow = follow_p.get(kind, 1.0)
            if random.Random(f"{cfg['seed']}-follow-{oid}-{mid}").random() \
                    < p_follow:
                await agent.env.action.follow(mid)

    start = date.fromisoformat(cfg["start_date"])
    n_days = (date.fromisoformat(cfg["end_date"]) - start).days + 1
    sem = asyncio.Semaphore(cfg["max_concurrency"])
    agent_days = run_dir / "agent_days.jsonl"

    async def agent_day(oid, agent, day_idx, day):
        async with sem:
            t0 = time.time()
            result = await agent.perform_action_by_llm()
            status, calls, error = classify_llm_result(result)
            t1 = time.time()
            try:
                interview = await agent.perform_interview(INTERVIEW_PROMPT)
                raw = interview.get("content")
                parsed = parse_interview(raw)
                iv_status = "ok" if parsed["ok"] else "parse_failed"
                iv_error = parsed["error"]
            except Exception as e:  # counted, not retried
                raw, parsed = None, parse_interview(None)
                iv_status = error_status(e)
                iv_error = f"{type(e).__name__}: {e}"
        write_jsonl(agent_days, {
            "day_index": day_idx, "date": day.isoformat(),
            "agent_id": id_map[oid]["agent_id"], "oasis_id": oid,
            "action_status": status, "tool_calls": calls,
            "action_error": error, "interview_status": iv_status,
            "interview_raw": raw, "interview_ok": parsed["ok"],
            "interview_error": iv_error, "stance": parsed["stance"],
            "intent": parsed["intent"], "reason": parsed["reason"],
            "action_s": round(t1 - t0, 1),
            "interview_s": round(time.time() - t1, 1)})
        return status, parsed["ok"]

    for day_idx in range(n_days):
        day = start + timedelta(days=day_idx)
        clock.time_step = day_idx
        for ev in timeline["events"]:
            ev_date = date.fromisoformat(ev["date"])
            if ev_date == day or (day_idx == 0 and ev_date < start):
                media = graph.get_agent(media_ids[ev["source_account"]])
                res = await media.env.action.create_post(ev["text"])
                write_jsonl(run_dir / "events_injected.jsonl", {
                    "day_index": day_idx, "date": day.isoformat(),
                    "event_id": ev["event_id"], "event_date": ev["date"],
                    "post_id": res.get("post_id"),
                    "success": res.get("success")})
        await platform.update_rec_table()
        t_day = time.time()
        results = await asyncio.gather(*[
            agent_day(oid, agent, day_idx, day)
            for oid, agent in persona_agents])
        n_ok = sum(1 for s, _ in results if s == "ok")
        n_iv = sum(1 for _, ok in results if ok)
        print(f"[{day}] day {day_idx}: {n_ok}/{len(results)} valid actions, "
              f"{n_iv}/{len(results)} parsed interviews, "
              f"{time.time() - t_day:.0f}s", flush=True)

    await env.close()
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
    from kanuni.export import export_run
    summary = export_run(run_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
