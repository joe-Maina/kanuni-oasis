"""Score simulation runs against what actually happened (Finance Bill 2024).

Reads the shared output format (output.jsonl), so it works for OASIS and
AgentSociety runs alike, and prints a scorecard: one score from 0 to 1 per
target, a combined similarity (the mean of the targets that could be
scored), and, for several runs (seeds), the mean and spread.

Targets (see ke_fb24_reference.json):
  stance        1 - |simulated share opposed - polled share opposed|, at the
                run's first day.
  geography     For one protest day: balanced accuracy of "did this county
                protest?" over the counties that have personas in the run.
                A county counts as protesting if any of its agents joined a
                protest (AgentSociety: join_protest action; OASIS has no
                offline action, so its stated intent attend_protest is used
                and the score is labelled as intent-based).
  escalation    Share of day pairs (18 vs 20 Jun, 20 vs 25 Jun, 18 vs 25 Jun)
                where more counties protested on the later day, as happened.

A similarity score is only meaningful for runs with a model whose training
cutoff is before June 2024; otherwise the model may simply remember what
happened. The scorecard prints the run's model cutoff next to the score.

Usage: python score.py RUN_DIR [RUN_DIR ...] [--reference ke_fb24_reference.json]
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_rows(run_dir: Path) -> list[dict]:
    return [json.loads(l) for l in (run_dir / "output.jsonl").read_text()
            .splitlines() if l.strip()]


def protested(row: dict) -> bool:
    if row["simulator"] == "agentsociety":
        return "join_protest" in row["actions"]
    return row["intent"] == "attend_protest"


def score_run(run_dir: Path, ref: dict) -> dict:
    rows = load_rows(run_dir)
    cfg = json.loads((run_dir / "run_config.json").read_text())
    t = ref["targets"]
    out = {"run": run_dir.name, "simulator": rows[0]["simulator"],
           "model": cfg["model"]["name"],
           "model_cutoff": cfg.get("model_cutoff"), "scores": {},
           "details": {}}

    first_day = min(r["day"] for r in rows)
    stances = [r["stance"] for r in rows
               if r["day"] == first_day and r["stance"] is not None]
    if stances:
        sim = sum(s < 0 for s in stances) / len(stances)
        real = t["stance_pre_amendment"]["share_oppose"]
        out["scores"]["stance"] = round(1 - abs(sim - real), 3)
        out["details"]["stance"] = {"day": first_day, "simulated": round(
            sim, 3), "real": real, "n": len(stances)}

    counties_in_run = {r["county"] for r in rows}
    by_day = defaultdict(set)
    for r in rows:
        if protested(r):
            by_day[r["day"]].add(r["county"])
    basis = "action" if out["simulator"] == "agentsociety" else "intent"
    for day, target in t["protest_counties"]["by_date"].items():
        real = target.get("counties")
        if not real or day not in {r["day"] for r in rows}:
            continue
        real = set(real) & counties_in_run
        sim = by_day[day]
        pos = [c for c in counties_in_run if c in real]
        neg = [c for c in counties_in_run if c not in real]
        tpr = sum(c in sim for c in pos) / len(pos) if pos else None
        tnr = sum(c not in sim for c in neg) / len(neg) if neg else None
        parts = [x for x in (tpr, tnr) if x is not None]
        out["scores"][f"geography_{day}"] = round(sum(parts) / len(parts), 3)
        out["details"][f"geography_{day}"] = {
            "basis": basis, "counties_in_run": len(counties_in_run),
            "real_protest_counties_in_run": sorted(real),
            "simulated_protest_counties": sorted(sim),
            "hit_rate": tpr, "correct_rejection_rate": tnr}

    order = [d for d in t["protest_escalation"]["order"]
             if d in {r["day"] for r in rows}]
    pairs = [(a, b) for i, a in enumerate(order) for b in order[i + 1:]]
    if pairs:
        good = sum(len(by_day[b]) > len(by_day[a]) for a, b in pairs)
        out["scores"]["escalation"] = round(good / len(pairs), 3)
        out["details"]["escalation"] = {
            "basis": basis,
            "counties_protesting": {d: len(by_day[d]) for d in order}}

    s = out["scores"]
    out["similarity"] = round(sum(s.values()) / len(s), 3) if s else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--reference", default=str(HERE / "ke_fb24_reference.json"))
    a = ap.parse_args()
    ref = json.loads(Path(a.reference).read_text())
    results = [score_run(Path(r), ref) for r in a.runs]
    report = {"runs": results}
    sims = [r["similarity"] for r in results if r["similarity"] is not None]
    if len(sims) > 1:
        report["similarity_mean"] = round(statistics.mean(sims), 3)
        report["similarity_sd"] = round(statistics.stdev(sims), 3)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
