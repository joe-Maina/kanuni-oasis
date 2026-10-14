"""Convert a Kanuni AS2 run folder into the shared output plus AS2-only files.

Writes:
  - output.jsonl: shared format, exactly SHARED_FIELDS, simulator "agentsociety".
    One row per agent per step; `day` is the step's first date.
  - household_days.jsonl: one row per agent per simulated day (money, work,
    missed payments, protest), AS2 only.
  - summary.json: stance per step, failure counts (model vs transport),
    what AS2 corrected on its own, protest turnout by county and day.

parse_failed is true when the agent's step did not end with the model's own
finish (error, network failure or AS2 fallback) or its stance answer was not
valid. Model and network failures are counted separately in summary.json.

Usage: python -m kanuni_as2.export kanuni_as2/runs/<run_id>
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

SHARED_FIELDS = [
    "run_id", "simulator", "seed", "condition", "day", "agent_id",
    "connectivity", "age_group", "county", "urban_rural", "livelihood",
    "stance", "intent", "actions", "post_text", "channel_exposures",
    "parse_failed",
]
WRITTEN = ("post_online", "post_in_whatsapp_group")
MODEL_FAILURES = ("llm_error", "no_valid_finish", "parse_failed")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def export_run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "run_config.json").read_text())
    personas_path = Path(cfg["personas"])
    if not personas_path.exists():  # run made on another machine
        personas_path = Path(__file__).resolve().parents[2] / "shared" / \
            personas_path.name
    personas = {p["agent_id"]: p
                for p in json.loads(personas_path.read_text())}
    recs = read_jsonl(run_dir / "agent_steps.jsonl")
    env_log = read_jsonl(run_dir / "env_log.jsonl")
    steps = read_jsonl(run_dir / "steps.jsonl")

    by_agent_day = defaultdict(lambda: {"exposures": [], "actions": []})
    household = []
    for row in env_log:
        if row["type"] == "household_day":
            household.append({k: v for k, v in row.items() if k != "type"})
            continue
        key = (row["agent_id"], row["date"])
        if row["type"] == "exposure":
            by_agent_day[key]["exposures"].append({
                "channel": row["channel"], "source": row["source"],
                "author": row["author"], "post_id": None,
                "event_id": row["event_id"]})
        else:
            by_agent_day[key]["actions"].append(row)

    out_rows = []
    for rec in sorted(recs, key=lambda r: (r["step_index"], r["agent_id"])):
        p = personas[rec["agent_id"]]
        start = date.fromisoformat(rec["date"])
        dates = [(start + timedelta(days=k)).isoformat()
                 for k in range(rec["days"])]
        acts = [a for d in dates
                for a in by_agent_day[(rec["agent_id"], d)]["actions"]]
        exposures = [e for d in dates
                     for e in by_agent_day[(rec["agent_id"], d)]["exposures"]]
        texts = [a["text"] for a in acts if a["action"] in WRITTEN
                 and a.get("text")]
        row = {
            "run_id": cfg["run_id"],
            "simulator": "agentsociety",
            "seed": cfg["seed"],
            "condition": cfg["condition"],
            "day": rec["date"],
            "agent_id": rec["agent_id"],
            "connectivity": p["connectivity"],
            "age_group": p["age_group"],
            "county": p["county"],
            "urban_rural": p["urban_rural"],
            "livelihood": p["livelihood"],
            "stance": rec["stance"],
            "intent": rec["intent"],
            "actions": [a["action"] for a in acts],
            "post_text": "\n\n".join(texts) or None,
            "channel_exposures": exposures,
            "parse_failed": rec["action_status"] != "ok"
            or rec["interview_status"] != "ok",
        }
        assert list(row) == SHARED_FIELDS
        out_rows.append(row)

    with open(run_dir / "output.jsonl", "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(run_dir / "household_days.jsonl", "w", encoding="utf-8") as f:
        for row in household:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {"run_id": cfg["run_id"], "simulator": "agentsociety",
               "scenario": Path(cfg["scenario"]).stem,
               "model": cfg["model"]["name"],
               "model_cutoff": cfg["model_cutoff"], "steps": []}
    recs_by_step = defaultdict(list)
    for r in recs:
        recs_by_step[r["date"]].append(r)
    rows_by_step = defaultdict(list)
    for row in out_rows:
        rows_by_step[row["day"]].append(row)
    hh_by_date = defaultdict(list)
    for h in household:
        hh_by_date[h["date"]].append(h)
    for day, rows in sorted(rows_by_step.items()):
        rs = recs_by_step[day]
        stances = [r["stance"] for r in rows if r["stance"] is not None]
        n = len(stances)
        actions = Counter(a for r in rows for a in r["actions"])
        summary["steps"].append({
            "day": day,
            "agents": len(rows),
            "stance_n": n,
            "stance_mean": round(sum(stances) / n, 2) if n else None,
            "stance_counts": {s: stances.count(s) for s in range(-2, 3)},
            "share_oppose": round(sum(s < 0 for s in stances) / n, 2)
            if n else None,
            "intent_counts": dict(Counter(r["intent"] for r in rows
                                          if r["intent"])),
            "action_counts": dict(actions),
            "action_status": dict(Counter(r["action_status"] for r in rs)),
            "interview_status": dict(Counter(r["interview_status"]
                                             for r in rs)),
            "heard_any_event": sum(any(e["event_id"]
                                       for e in r["channel_exposures"])
                                   for r in rows),
        })

    protests = Counter()
    for row in env_log:
        if row["type"] == "action" and row["action"] == "join_protest":
            protests[(row["date"], row["county"])] += 1
    summary["protest_turnout"] = [
        {"date": d, "county": c, "agents": k}
        for (d, c), k in sorted(protests.items())]
    summary["household"] = [{
        "date": d,
        "agents": len(hs),
        "mean_income": round(sum(h["income"] for h in hs) / len(hs)),
        "mean_spent": round(sum(h["spent_total"] for h in hs) / len(hs)),
        "missed_any_payment": sum(bool(h["missed"]) for h in hs),
        "cutting_back": sum(h["spending_plan"] != "normal" for h in hs),
        "protested": sum(h["protested"] for h in hs),
        "in_debt": sum(h["debt_end"] > 0 for h in hs),
    } for d, hs in sorted(hh_by_date.items())]

    def total(field):
        return sum(r.get("step_counts", {}).get(field, 0)
                   + r.get("interview_counts", {}).get(field, 0)
                   for r in recs)
    summary["totals"] = {
        "agent_steps": len(out_rows),
        "rows_parse_failed": sum(r["parse_failed"] for r in out_rows),
        "action_status": dict(Counter(r["action_status"] for r in recs)),
        "interview_status": dict(Counter(r["interview_status"]
                                         for r in recs)),
        "model_action_failures": sum(r["action_status"] in MODEL_FAILURES
                                     for r in recs),
        "model_interview_failures": sum(
            r["interview_status"] in MODEL_FAILURES for r in recs),
        "infra_failures": sum(r["action_status"] == "infra_error"
                              or r["interview_status"] == "infra_error"
                              for r in recs),
        "context_overflows": sum(r["action_status"] == "context_overflow"
                                 or r["interview_status"] == "context_overflow"
                                 for r in recs),
        # What AS2 fixed by itself; reported so nothing is hidden.
        "as2_malformed_model_turns": total("malformed_turns"),
        "as2_transport_retries": sum(s["transport_retries"] for s in steps),
        "agent_llm_calls": total("llm_calls"),
        "env_requests": total("env_requests"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    print(json.dumps(export_run(Path(ap.parse_args().run_dir)), indent=2))
