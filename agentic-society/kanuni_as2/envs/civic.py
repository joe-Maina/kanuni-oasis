"""Offline political action: protests, contacting the MP, community meetings.

Agents are told only what people knew at the time: public calls to protest
(from the scenario, the same for everyone) and protests in their own county
that other simulated agents joined the step before. They are never told
where real protests happened, so simulated turnout by county and day can be
compared with real records (validation) without leaking them. Joining costs
that day's work income (the household module reads protested()).
"""
from __future__ import annotations

from datetime import date, datetime

from agentsociety2.env import EnvBase, tool

from kanuni_as2.common import World, append_jsonl, check_agent

MEETINGS = ("church_or_mosque", "chama", "chief_baraza", "workplace")


class KanuniCivicEnv(EnvBase):
    """Civic life: protests, contacting your MP, and community meetings."""

    def __init__(self, world: World, log_path: str):
        super().__init__()
        self.world = world
        self.log_path = log_path
        self._protested: set[tuple[int, date]] = set()
        self._last_step_day: date | None = None

    @classmethod
    def description(cls) -> str:
        return "Civic action: join a protest, contact your MP, attend a meeting."

    def protested(self, agent_id: int, day: date) -> bool:
        return (agent_id, day) in self._protested

    def observe_for(self, agent_id: int) -> str:
        day = self.t.date()
        county = self.world.personas[agent_id]["county"]
        lines = []
        call = self.world.protest_call(day)
        if call:
            lines.append(call["text"])
        if self._last_step_day is not None:
            n = sum(1 for a, d in self._protested
                    if d == self._last_step_day and a != agent_id
                    and self.world.personas[a]["county"] == county)
            if n:
                lines.append(f"Last time, people in {county} joined protests "
                             f"against the bill ({n} that you know of).")
        if lines:
            lines.append("Joining a protest today means not working today.")
        return " ".join(lines)

    @tool(readonly=False)
    async def join_protest(self, agent_id: int) -> str:
        """Join today's protest against the bill in your area (you will not work today).

        :param agent_id: Your own agent id.
        :returns: Confirmation.
        """
        check_agent(agent_id)
        day = self.t.date()
        county = self.world.personas[agent_id]["county"]
        self._protested.add((agent_id, day))
        self._log("join_protest", agent_id, county=county,
                  called_today=self.world.protest_call(day) is not None)
        return "You joined the protest today."

    @tool(readonly=False)
    async def contact_mp(self, agent_id: int, message: str) -> str:
        """Call, text or visit your Member of Parliament about the bill.

        :param agent_id: Your own agent id.
        :param message: What you tell the MP.
        :returns: Confirmation.
        """
        check_agent(agent_id)
        self._log("contact_mp", agent_id, text=message)
        return "Your message was delivered to the MP's office."

    @tool(readonly=False)
    async def attend_meeting(self, agent_id: int, meeting: str,
                             what_you_said: str) -> str:
        """Attend a community meeting and speak about the bill.

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
        self._last_step_day = t.date()
