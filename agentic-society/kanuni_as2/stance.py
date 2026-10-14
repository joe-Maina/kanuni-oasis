"""Daily stance/intent interview: prompt and strict parser.

The parser does not guess. Anything that is not a JSON object with a valid
stance and intent counts as a parse failure, so failure rates are comparable
across models.
"""
import json
import re

INTENTS = ("none", "post", "attend_protest", "contact_mp")

# Same question as camel-oasis/oasis/kanuni/stance.py; only the bill name is
# a parameter, so stance numbers from both simulators stay comparable.
INTERVIEW_TEMPLATE = (
    "This is a private survey question, not a post. Based only on what you "
    "have seen so far, what is your current position on the {bill}, and "
    "what, if anything, do you intend to do about it?\n"
    "Reply with only a JSON object and nothing else, in this form:\n"
    '{{"stance": <integer from -2 (strongly oppose) to 2 (strongly support)>, '
    '"intent": "<one of: none, post, attend_protest, contact_mp>", '
    '"reason": "<one short sentence>"}}')


def interview_prompt(bill: str) -> str:
    return INTERVIEW_TEMPLATE.format(bill=bill)


INTERVIEW_PROMPT = interview_prompt("Finance Bill 2024")

_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_interview(text: str | None) -> dict:
    """Return {"ok", "stance", "intent", "reason", "error"}."""
    out = {"ok": False, "stance": None, "intent": None, "reason": None,
           "error": None}
    if not text:
        out["error"] = "empty response"
        return out
    body = _THINK.sub("", text).strip()
    fence = _FENCE.search(body)
    if fence:
        body = fence.group(1).strip()
    start, end = body.find("{"), body.rfind("}")
    if start == -1 or end <= start:
        out["error"] = "no JSON object"
        return out
    try:
        obj = json.loads(body[start:end + 1])
    except json.JSONDecodeError as e:
        out["error"] = f"invalid JSON: {e.msg}"
        return out
    stance = obj.get("stance")
    if isinstance(stance, str) and re.fullmatch(r"[+-]?\d", stance.strip()):
        stance = int(stance)
    if isinstance(stance, float) and stance.is_integer():
        stance = int(stance)
    if not isinstance(stance, int) or isinstance(stance, bool) \
            or not -2 <= stance <= 2:
        out["error"] = f"bad stance: {obj.get('stance')!r}"
        return out
    intent = obj.get("intent")
    if isinstance(intent, str):
        intent = intent.strip().lower().replace(" ", "_").replace("-", "_")
    if intent not in INTENTS:
        out["error"] = f"bad intent: {obj.get('intent')!r}"
        return out
    reason = obj.get("reason")
    out.update(ok=True, stance=stance, intent=intent,
               reason=reason if isinstance(reason, str) else None)
    return out
