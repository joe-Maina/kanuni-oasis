"""Acting on an uploaded policy: the actions its outcome plan gives each person.

Used in place of KanuniCivicEnv when the run has an outcome plan from the
Kanuni API (config "outcomes"). The Finance Bill's fixed actions (join a
protest, contact the MP) become the plan's behaviours for the person's
stakeholder category, e.g. "Send polling-station results within two hours"
for a presiding officer. Each person is shown their own list in the daily
observation and acts through one tool, take_action, which refuses an action
that isn't on their list. The observation also lists what they have already
done about the policy, and says acting is optional: shown the same list
every morning and nothing else, agents took the same action every day. As in KanuniCivicEnv, protesting costs that day's
work income (the household module reads protested()).
"""
from __future__ import annotations

from datetime import date, datetime

from agentsociety2.env import EnvBase, tool

from kanuni_as2.common import World, append_jsonl, check_agent
from kanuni_as2.envs.civic import MEETINGS
from kanuni_as2.survey import NO_CHANGE, SurveyPlan

PROTEST = "protest"
RECENT_ACTIONS = 5  # past actions shown in the observation


class KanuniPolicyEnv(EnvBase):
    """Acting on the policy: the actions open to you, and community meetings."""

    def __init__(self, world: World, log_path: str, survey: SurveyPlan,
                 bill: str):
        super().__init__()
        self.world = world
        self.log_path = log_path
        self.survey = survey
        self.bill = bill
        self._protested: set[tuple[int, date]] = set()
        self._done: dict[int, list[tuple[date, str]]] = {}

    @classmethod
    def description(cls) -> str:
        return ("Acting on the policy: take one of the actions open to you "
                "about it, or speak at a community meeting.")

    def options_for(self, agent_id: int) -> list[dict]:
        cat = self.survey.category_of(self.world.personas[agent_id])
        return [b for b in self.survey.behaviours_for(cat)
                if b["id"] != NO_CHANGE]

    def protested(self, agent_id: int, day: date) -> bool:
        return (agent_id, day) in self._protested

    def observe_for(self, agent_id: int) -> str:
        options = self.options_for(agent_id)
        if not options:
            return ""
        lines = [f"Things you could do about the {self.bill}, if anything "
                 "today gives you a reason to (use take_action with the id):"]
        lines += [f"- {b['id']}: {b['label']}" for b in options]
        if any(b["id"] == PROTEST for b in options):
            lines.append("Joining a protest today means not working today.")
        done = self._done.get(agent_id)
        if done:
            labels = {b["id"]: b["label"] for b in options}
            lines.append("What you have already done about it:")
            lines += [f"- {d.isoformat()}: {labels.get(a, a)}"
                      for d, a in done[-RECENT_ACTIONS:]]
        return "\n".join(lines)

    @tool(readonly=False)
    async def take_action(self, agent_id: int, action: str,
                          what_you_did: str) -> str:
        """Do one of the things open to you about the policy (listed in your observation).

        :param agent_id: Your own agent id.
        :param action: The id of the action, from your list.
        :param what_you_did: What you did, in one or two sentences.
        :returns: Confirmation.
        """
        check_agent(agent_id)
        ids = [b["id"] for b in self.options_for(agent_id)]
        if action not in ids:
            raise ValueError(f"action must be one of {ids}")
        day = self.t.date()
        if action == PROTEST:
            self._protested.add((agent_id, day))
        self._done.setdefault(agent_id, []).append((day, action))
        self._log("take_action", agent_id, policy_action=action,
                  county=self.world.personas[agent_id]["county"],
                  text=what_you_did)
        return "Done."

    @tool(readonly=False)
    async def attend_meeting(self, agent_id: int, meeting: str,
                             what_you_said: str) -> str:
        """Attend a community meeting and speak about the policy.

        :param agent_id: Your own agent id.
        :param meeting: One of "church_or_mosque", "chama", "chief_baraza", "workplace".
        :param what_you_said: What you said there.
        :returns: Confirmation.
        """
        check_agent(agent_id)
        if meeting not in MEETINGS:
            raise ValueError(f"meeting must be one of {MEETINGS}")
        self._log("attend_meeting", agent_id, meeting=meeting,
                  text=what_you_said)
        return "Noted."

    def _log(self, action: str, agent_id: int, **extra) -> None:
        append_jsonl(self.log_path, {
            "type": "action", "date": self.t.date().isoformat(),
            "agent_id": agent_id, "action": action, **extra})

    async def step(self, tick: int, t: datetime):
        pass
