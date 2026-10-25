"""Convert a Kanuni OASIS run folder into the shared output format.

Reads oasis.db (trace/post tables), agent_days.jsonl (stance, intent and
LLM status from the runner), id_map.json and the persona file, and writes:
  - output.jsonl: one row per persona agent per simulated day, with exactly
    the shared fields (see SHARED_FIELDS) followed by OUTCOME_FIELDS;
  - summary.json: per-day stance summary and failure counts, and with an
    outcome plan, per-day actions by category, wellbeing and issue answers.

With an outcome plan, "intent" holds the action the agent chose from its
category's list (kanuni/survey.py) instead of one of stance.py's INTENTS.

parse_failed is true when the agent's LLM turn produced no valid action
(no tool call, model error, or every tool call rejected by the platform) or
when its stance interview could not be parsed, including when a network
error meant there was no answer at all. summary.json separates model
failures from network (infra) failures.

Usage: python -m kanuni.export kanuni/runs/<run_id>
"""
import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

from kanuni.survey import summarise

SHARED_FIELDS = [
    "run_id", "simulator", "seed", "condition", "day", "agent_id",
    "connectivity", "age_group", "county", "urban_rural", "livelihood",
    "stance", "intent", "actions", "post_text", "channel_exposures",
    "parse_failed",
]
# After the shared fields: what the policy survey adds. Empty or null on
# runs without an outcome plan, except the category when the persona has one.
OUTCOME_FIELDS = ["stakeholder_category", "phase", "wellbeing", "satisfaction",
                  "reason", "answers", "answer_errors"]
MODEL_ACTION_FAILURES = ("no_tool_call", "llm_error", "action_rejected")
# Trace rows that are bookkeeping, not agent choices.
NOT_ACTIONS = {"refresh", "interview", "sign_up"}


INFRA_NAMES = ("APIConnectionError", "APITimeoutError", "InternalServerError",
               "APIStatusError", "ConnectError")


def normalize(rec: dict) -> dict:
    """Bring records from older runner versions up to the current fields.

    Older records kept a single `error` field that the interview error could
    overwrite, and did not separate network failures from model failures.
    A network error there is assumed to have hit the action as well.
    """
    if "interview_status" in rec:
        return rec
    err = rec.get("error") or ""
    infra = any(n in err for n in INFRA_NAMES)
    if rec["action_status"] == "llm_error" and infra:
        rec["action_status"] = "infra_error"
    if rec["interview_ok"]:
        rec["interview_status"] = "ok"
    elif err.startswith("interview"):
        rec["interview_status"] = "infra_error" if infra else "llm_error"
    else:
        rec["interview_status"] = "parse_failed"
    return rec


SHARED = Path(__file__).resolve().parents[3] / "shared"


def local_path(path: str) -> Path:
    """A shared-input path from a run config, or the same file in this
    checkout's shared/ folder when the run was made on another machine."""
    p = Path(path)
    return p if p.exists() else SHARED / p.name


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def export_run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "run_config.json").read_text())
    id_map = {int(k): v for k, v in
              json.loads((run_dir / "id_map.json").read_text()).items()}
    personas = {p["agent_id"]: p
                for p in json.loads(local_path(cfg["personas"]).read_text())}
    days = {(r["day_index"], r["agent_id"]): normalize(r)
            for r in read_jsonl(run_dir / "agent_days.jsonl")}
    event_posts = {r["post_id"]: r["event_id"]
                   for r in read_jsonl(run_dir / "events_injected.jsonl")}
    start = date.fromisoformat(cfg["start_date"])

    con = sqlite3.connect(run_dir / "oasis.db")
    quotes = dict(con.execute(
        "SELECT post_id, quote_content FROM post "
        "WHERE quote_content IS NOT NULL").fetchall())
    trace = defaultdict(list)
    for user_id, created_at, action, info in con.execute(
            "SELECT user_id, created_at, action, info FROM trace"):
        try:
            day_idx = int(created_at)
        except (TypeError, ValueError):
            continue
        if day_idx < 0 or id_map.get(user_id, {}).get("kind") != "persona":
            continue
        trace[(day_idx, id_map[user_id]["agent_id"])].append(
            (action, json.loads(info) if info else {}))
    con.close()

    def author(oasis_user_id):
        m = id_map.get(oasis_user_id, {})
        return m.get("account_id") if m.get("kind") == "media" else (
            m.get("agent_id"))

    out_rows = []
    for (day_idx, agent_id), rec in sorted(days.items()):
        p = personas[agent_id]
        actions, texts, exposures, seen = [], [], [], set()
        for action, info in trace.get((day_idx, agent_id), []):
            if action == "refresh":
                for post in info.get("posts", []):
                    if post["post_id"] in seen:
                        continue
                    seen.add(post["post_id"])
                    acct = id_map.get(post["user_id"], {})
                    exposures.append({
                        "channel": "oasis_feed",
                        # media (news accounts), activist, influencer,
                        # pro_government, or peer for persona agents' posts.
                        "source": "peer" if acct.get("kind") != "media"
                        else {"news": "media"}.get(
                            acct.get("account_kind", "news"),
                            acct.get("account_kind")),
                        "author": author(post["user_id"]),
                        "post_id": post["post_id"],
                        "event_id": event_posts.get(post["post_id"]),
                    })
                continue
            if action in NOT_ACTIONS:
                continue
            actions.append(action)
            if action in ("create_post", "create_comment"):
                texts.append(info.get("content", ""))
            elif action == "quote_post":
                texts.append(quotes.get(info.get("new_post_id"), ""))
        failed = rec["action_status"] != "ok" or not rec["interview_ok"]
        row = {
            "run_id": cfg["run_id"],
            "simulator": "oasis",
            "seed": cfg["seed"],
            "condition": cfg["condition"],
            "day": (start + timedelta(days=day_idx)).isoformat(),
            "agent_id": agent_id,
            "connectivity": p["connectivity"],
            "age_group": p["age_group"],
            "county": p["county"],
            "urban_rural": p["urban_rural"],
            "livelihood": p["livelihood"],
            "stance": rec["stance"],
            "intent": rec["intent"],
            "actions": actions,
            "post_text": "\n\n".join(t for t in texts if t) or None,
            "channel_exposures": exposures,
            "parse_failed": failed,
            "stakeholder_category": rec.get("stakeholder_category",
                                            p.get("stakeholder_category")),
            "phase": rec.get("phase"),
            "wellbeing": rec.get("wellbeing"),
            "satisfaction": rec.get("satisfaction"),
            "reason": rec.get("reason"),
            "answers": rec.get("answers") or {},
            "answer_errors": rec.get("answer_errors") or {},
        }
        assert list(row) == SHARED_FIELDS + OUTCOME_FIELDS
        out_rows.append(row)

    with open(run_dir / "output.jsonl", "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {"run_id": cfg["run_id"], "model": cfg["model"]["name"],
               "model_cutoff": cfg["model_cutoff"], "days": []}
    by_day = defaultdict(list)
    for row in out_rows:
        by_day[row["day"]].append(row)
    recs_by_day = defaultdict(list)
    for (day_idx, _), rec in days.items():
        recs_by_day[rec["date"]].append(rec)
    for day, rows in sorted(by_day.items()):
        stances = [r["stance"] for r in rows if r["stance"] is not None]
        status = Counter(r["action_status"] for r in recs_by_day[day])
        n = len(stances)
        summary["days"].append({
            "day": day,
            "events": sorted({e["event_id"] for r in rows
                              for e in r["channel_exposures"]
                              if e["event_id"]}),
            "agents": len(rows),
            "stance_n": n,
            "stance_mean": round(sum(stances) / n, 2) if n else None,
            "stance_counts": {s: stances.count(s) for s in range(-2, 3)},
            "share_oppose": round(sum(s < 0 for s in stances) / n, 2)
            if n else None,
            "share_support": round(sum(s > 0 for s in stances) / n, 2)
            if n else None,
            "intent_counts": dict(Counter(r["intent"] for r in rows
                                          if r["intent"])),
            "action_status": dict(status),
            "interview_status": dict(Counter(
                r["interview_status"] for r in recs_by_day[day])),
            "rows_parse_failed": sum(r["parse_failed"] for r in rows),
            "posts_written": sum(r["post_text"] is not None for r in rows),
            "saw_any_event": sum(any(e["event_id"]
                                     for e in r["channel_exposures"])
                                 for r in rows),
        })
    recs = list(days.values())
    model_action_fail = sum(r["action_status"] in MODEL_ACTION_FAILURES
                            for r in recs)
    model_iv_fail = sum(r["interview_status"] in ("parse_failed",
                                                  "llm_error")
                        for r in recs)
    summary["totals"] = {
        "agent_days": len(out_rows),
        "rows_parse_failed": sum(r["parse_failed"] for r in out_rows),
        "action_status": dict(Counter(r["action_status"] for r in recs)),
        "interview_status": dict(Counter(r["interview_status"]
                                         for r in recs)),
        # Failures attributable to the model, excluding network errors.
        "model_action_failures": model_action_fail,
        "model_interview_failures": model_iv_fail,
        # Answers that parsed only after stance.py added a missing "}".
        "interviews_repaired": sum(bool(r.get("interview_repaired"))
                                   for r in recs),
        "infra_failures": sum(r["action_status"] == "infra_error"
                              or r["interview_status"] == "infra_error"
                              for r in recs),
        # Prompts the shim refused as too long for the context window.
        "context_overflows": sum(r["action_status"] == "context_overflow"
                                 or r["interview_status"] == "context_overflow"
                                 for r in recs),
    }
    plan = None
    if cfg.get("outcomes"):
        plan_path = Path(cfg["outcomes"])
        if not plan_path.exists():  # run made on another machine
            plan_path = run_dir / plan_path.name
        plan = json.loads(plan_path.read_text())
    summary["outcomes"] = summarise(out_rows, plan)
    summary["totals"]["issue_answers_failed"] = sum(
        len(r["answer_errors"]) for r in out_rows)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    print(json.dumps(export_run(Path(ap.parse_args().run_dir)), indent=2))
