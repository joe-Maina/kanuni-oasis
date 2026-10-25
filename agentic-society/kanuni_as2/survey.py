"""Policy survey from the backend's outcome plan: prompt, strict parser and
summary.

The Kanuni API writes an outcome plan for each uploaded policy (see
kanuni_backend FRAMEWORK.md, stage 6): the behaviours each stakeholder
category may choose, and the issues each category is asked about in each
phase. A run given the plan (config key "outcomes") asks every agent, each
step:
  - stance on the policy, -2..2, as in stance.py, so stance stays
    comparable with the Finance Bill runs;
  - wellbeing, -2..2: how the policy has left their daily life so far;
  - satisfaction, 0..10: how satisfied they are with their life right
    now, whatever the cause. A level, not a change, so it can be set
    against the persona's baseline satisfaction (0-100) and followed day
    by day with more range than wellbeing's five points;
  - action: what they did about it today, one of their category's
    behaviour ids, or no_change for nothing new today (this fills the
    shared "intent" field). Asking for today rather than an intention
    keeps the answer from repeating a fixed position every day, so the
    counts show what people do over time;
  - reason, one sentence;
  - answers to at most a few issue questions for their category and the
    run's phase (pilot for the first steps, then steady_state). An issue
    narrowed to some stakeholder groups (asked_of_groups) is asked only
    of people in them (the persona's group_id).

The parser does not guess, like stance.py, and makes the same one repair
(a dropped closing brace, which the nested answers object can hide from
stance.py's check). The interview counts as ok only
when stance, wellbeing, satisfaction and action are valid; each issue answer is checked
on its own and a bad or missing one is recorded in answer_errors, so issue
failure rates are reported, not hidden.

camel-oasis/oasis/kanuni/survey.py and agentic-society/kanuni_as2/survey.py
are the same file; keep them identical.
"""
from collections import Counter, defaultdict

from .stance import extract_json, small_int

NO_CHANGE = "no_change"
PHASES = ("pilot", "steady_state")
FALLBACK_CATEGORY = "public"
SCALE_HINT = "an integer from 1 (not at all) to 5 (completely)"


class SurveyPlan:
    """One outcome plan, looked up by category and phase."""

    def __init__(self, plan: dict):
        self.plan = plan
        self.behaviours = {c: list(bs) for c, bs in plan["behaviours"].items()}
        self.issues = list(plan.get("issues") or [])

    def category_of(self, persona: dict) -> str:
        """The persona's category, or the public's when the plan has no
        behaviours for it (personas built before categories existed)."""
        c = persona.get("stakeholder_category")
        return c if c in self.behaviours else FALLBACK_CATEGORY

    def behaviours_for(self, category: str) -> list[dict]:
        return self.behaviours.get(category) or [
            {"id": NO_CHANGE, "label": "Carry on as before"}]

    def questions_for(self, category: str, phase: str,
                      group: str | None = None) -> list[dict]:
        return [i for i in self.issues
                if category in i["asked_of"] and i["phase"] == phase
                and (not i.get("asked_of_groups") or group in i["asked_of_groups"])]

    def prompt(self, bill: str, category: str, phase: str,
               group: str | None = None) -> str:
        return survey_prompt(bill, self.behaviours_for(category),
                             self.questions_for(category, phase, group))

    def parse(self, text: str | None, category: str, phase: str,
              group: str | None = None) -> dict:
        return parse_survey(text, self.behaviours_for(category),
                            self.questions_for(category, phase, group))


def phase_for(step_index: int, n_steps: int, pilot_steps: int | None = None) -> str:
    """Pilot for the first quarter of the run (at least one step), unless
    the config sets pilot_steps; steady_state after."""
    if pilot_steps is None:
        pilot_steps = max(1, n_steps // 4)
    return "pilot" if step_index < pilot_steps else "steady_state"


def _answer_hint(q: dict) -> str:
    if q["answer_type"] == "yes_no":
        return "true or false"
    if q["answer_type"] == "scale_1_5":
        return SCALE_HINT
    return "one of: " + ", ".join(f'"{o}"' for o in q["options"])


def survey_prompt(bill: str, behaviours: list[dict], questions: list[dict]) -> str:
    lines = [
        "This is a private survey question, not a post. Based only on what "
        f"you have seen and lived so far, answer about the {bill}.",
        "Reply with only a JSON object and nothing else, in this form:",
        '{"stance": <integer from -2 (strongly oppose) to 2 (strongly support)>, '
        '"wellbeing": <integer from -2 (it has left you much worse off) to 2 '
        '(much better off); 0 if it has not changed your life>, '
        '"satisfaction": <integer from 0 (not at all satisfied) to 10 (completely '
        'satisfied): how satisfied you are with your life right now>, '
        '"action": "<what you did about it today: one id from the list below>", '
        '"reason": "<one short sentence>"'
        + (', "answers": {"<question id>": <answer>, ...}' if questions else "")
        + "}",
        "",
        "For action, answer for today only. Something you did on an earlier "
        "day counts only if you did it again today; if you did nothing new "
        f"about it today, answer {NO_CHANGE}.",
        "",
        "Actions (id: meaning):",
    ]
    lines += [f"- {b['id']}: {b['label']}" + (" (nothing new today)" if b["id"] == NO_CHANGE else "")
              for b in behaviours]
    if questions:
        lines += ["", "Questions (answer every one, by id):"]
        lines += [f"- {q['id']}: {q['question']} Answer {_answer_hint(q)}."
                  for q in questions]
    return "\n".join(lines)


def _answer(q: dict, value):
    """The answer in its canonical form, or None when it isn't valid."""
    if q["answer_type"] == "yes_no":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("yes", "no", "true", "false"):
            return value.strip().lower() in ("yes", "true")
        return None
    if q["answer_type"] == "scale_1_5":
        return small_int(value, 1, 5)
    if isinstance(value, str):
        by_text = {o.strip().lower(): o for o in q["options"]}
        return by_text.get(value.strip().lower())
    return None


def parse_survey(text: str | None, behaviours: list[dict], questions: list[dict]) -> dict:
    """Return {"ok", "stance", "wellbeing", "satisfaction", "intent", "reason", "answers",
    "answer_errors", "error", "repaired"}. "intent" is the chosen action id."""
    out = {"ok": False, "stance": None, "wellbeing": None, "satisfaction": None, "intent": None,
           "reason": None, "answers": {}, "answer_errors": {}, "error": None,
           "repaired": None}
    obj, error, repaired = extract_json(text)
    if obj is None and error.startswith("invalid JSON"):
        # The nested "answers" object hides a dropped final brace from
        # extract_json's repair, so one is tried here the same way.
        obj, _, _ = extract_json(text.rstrip() + "}")
        repaired = "added closing brace" if obj is not None else None
    if obj is None:
        out["error"] = error
        return out
    stance = small_int(obj.get("stance"), -2, 2)
    if stance is None:
        out["error"] = f"bad stance: {obj.get('stance')!r}"
        return out
    wellbeing = small_int(obj.get("wellbeing"), -2, 2)
    if wellbeing is None:
        out["error"] = f"bad wellbeing: {obj.get('wellbeing')!r}"
        return out
    satisfaction = small_int(obj.get("satisfaction"), 0, 10)
    if satisfaction is None:
        out["error"] = f"bad satisfaction: {obj.get('satisfaction')!r}"
        return out
    action = obj.get("action")
    if isinstance(action, str):
        action = action.strip().lower().replace(" ", "_").replace("-", "_")
    if action not in {b["id"] for b in behaviours}:
        out["error"] = f"bad action: {obj.get('action')!r}"
        return out
    given = obj.get("answers") if isinstance(obj.get("answers"), dict) else {}
    for q in questions:
        if q["id"] not in given:
            out["answer_errors"][q["id"]] = "missing"
            continue
        value = _answer(q, given[q["id"]])
        if value is None:
            out["answer_errors"][q["id"]] = f"bad answer: {given[q['id']]!r}"[:120]
        else:
            out["answers"][q["id"]] = value
    reason = obj.get("reason")
    out.update(ok=True, stance=stance, wellbeing=wellbeing, satisfaction=satisfaction, intent=action,
               reason=reason if isinstance(reason, str) else None,
               repaired=repaired)
    return out


def summarise(rows: list[dict], plan: dict | None) -> dict | None:
    """Per-period outcome counts for summary.json: actions by category,
    wellbeing, satisfaction, and each issue's answers with how many failed. Rows are the
    exported rows (one per agent per day or step)."""
    if not plan:
        return None
    issues = {i["id"]: i for i in plan.get("issues") or []}
    by_day = defaultdict(list)
    for r in rows:
        by_day[r["day"]].append(r)
    days = []
    for day, rs in sorted(by_day.items()):
        actions = defaultdict(Counter)
        for r in rs:
            if r["intent"]:
                actions[r["stakeholder_category"]][r["intent"]] += 1
        wb = [r["wellbeing"] for r in rs if r["wellbeing"] is not None]
        sat = [r.get("satisfaction") for r in rs if r.get("satisfaction") is not None]
        answers = {}
        for iid, issue in issues.items():
            asked = [r for r in rs if iid in r["answers"] or iid in r["answer_errors"]]
            if not asked:
                continue
            got = Counter(str(r["answers"][iid]).lower() if isinstance(r["answers"].get(iid), bool)
                          else str(r["answers"][iid]) for r in asked if iid in r["answers"])
            answers[iid] = {"question": issue["question"], "phase": issue["phase"],
                            "asked": len(asked), "failed": len(asked) - sum(got.values()),
                            "counts": dict(got)}
        days.append({"day": day,
                     "phase": next((r["phase"] for r in rs if r["phase"]), None),
                     "actions_by_category": {c: dict(n) for c, n in actions.items()},
                     "wellbeing_n": len(wb),
                     "wellbeing_mean": round(sum(wb) / len(wb), 2) if wb else None,
                     "satisfaction_n": len(sat),
                     "satisfaction_mean": round(sum(sat) / len(sat), 2) if sat else None,
                     "issues": answers})
    return {"plan_version": plan.get("version"), "days": days}
