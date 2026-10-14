"""How news about the bill reaches each persona: media channels and word of mouth.

Exposure is decided by code, not by the model: an event published today
reaches a persona through each channel it uses with that channel's reach
probability (channels.json). Messages agents pass to their ties arrive at the
next step. Every exposure and every message is logged, which is where the
shared output's channel_exposures and post_text come from.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime

from agentsociety2.env import EnvBase, tool

from kanuni_as2.common import World, append_jsonl, chance, check_agent

AUDIENCES = ("neighbours", "workmates", "family")
ONLINE_PLATFORMS = ("x_twitter", "tiktok", "facebook", "instagram",
                    "youtube", "linkedin")


class KanuniInformationEnv(EnvBase):
    """Delivers the day's news through each person's own channels, and lets
    people tell others what they think (in person, on WhatsApp or online)."""

    def __init__(self, world: World, log_path: str):
        super().__init__()
        self.world = world
        self.log_path = log_path
        # Last date whose news has been delivered; the next step delivers
        # everything after it, so weekly steps miss nothing.
        self.covered_until: date | None = None
        # Messages waiting to be delivered next step: agent -> [(from, via, text)]
        self._inbox: dict[int, list[tuple[int, str, str]]] = defaultdict(list)
        self._delivering: dict[int, list[tuple[int, str, str]]] = {}

    @classmethod
    def description(cls) -> str:
        return ("News and conversations: what each person heard today and "
                "tools to tell others what they think.")

    # ---- observation (called by code, not by the model) ----

    def observe_for(self, agent_id: int) -> str:
        """Today's news for one person. Deterministic; logs exposures."""
        day = self.t.date()
        chans = self.world.channels[agent_id]
        cfg = self.world.channels_cfg["channels"]
        weekday = day.strftime("%a")
        lines = []
        for ev in self.world.events_between(self.covered_until, day):
            acct = self.world.accounts.get(ev.get("source_account"), {})
            kind = acct.get("kind", "news")
            # Layer posts name the channels they travel on (social media and
            # WhatsApp forwards for activists); news reaches every channel.
            allowed = ev.get("channels")
            for ch in chans:
                if allowed is not None and ch not in allowed:
                    continue
                c = cfg[ch]
                if "days" in c and weekday not in c["days"]:
                    continue
                if not chance(c["reach"], self.world.seed, "reach", agent_id,
                              ev["event_id"], ch, day):
                    continue
                who = "" if kind == "news" else f"{acct['name']} posted: "
                lines.append(f"- Via {c['label']}: {who}{ev['text']}")
                append_jsonl(self.log_path, {
                    "type": "exposure", "date": day.isoformat(),
                    "agent_id": agent_id, "channel": ch,
                    "source": kind if kind != "news" else (
                        "media" if c["kind"] != "peer" else "peer"),
                    "event_id": ev["event_id"],
                    "author": None if kind == "news" else
                    ev["source_account"]})
                break  # one channel is enough to hear an event
        for sender, via, text in self._delivering.get(agent_id, []):
            name = self.world.personas[sender]["name"]
            lines.append(f"- {name} told you ({via}): {text}")
            append_jsonl(self.log_path, {
                "type": "exposure", "date": day.isoformat(),
                "agent_id": agent_id, "channel": via, "source": "peer",
                "event_id": None, "author": sender})
        if not lines:
            return "You heard nothing new about national politics today."
        return "What you heard today:\n" + "\n".join(lines)

    # ---- tools for the model ----

    @tool(readonly=False)
    async def tell_people(self, agent_id: int, audience: str,
                          message: str) -> str:
        """Talk in person to people you know about the bill or the economy.

        :param agent_id: Your own agent id.
        :param audience: One of "neighbours", "workmates", "family".
        :param message: What you say, in your own words (one or two sentences).
        :returns: Who will hear it.
        """
        check_agent(agent_id)
        if audience not in AUDIENCES:
            raise ValueError(f"audience must be one of {AUDIENCES}")
        targets = self.world.ties[agent_id][audience]
        for t in targets:
            self._inbox[t].append((agent_id, f"in person, {audience}", message))
        self._log_action(agent_id, "tell_people", message, audience=audience,
                         reached=len(targets))
        return f"{len(targets)} people heard you."

    @tool(readonly=False)
    async def post_in_whatsapp_group(self, agent_id: int, message: str) -> str:
        """Post a message in your family or community WhatsApp group.

        :param agent_id: Your own agent id.
        :param message: The message text.
        :returns: How many group members can read it.
        """
        check_agent(agent_id)
        if "whatsapp" not in self.world.channels[agent_id]:
            raise ValueError("you do not use WhatsApp")
        ties = self.world.ties[agent_id]
        targets = {t for t in ties["family"] + ties["neighbours"]
                   if "whatsapp" in self.world.channels[t]}
        for t in targets:
            self._inbox[t].append((agent_id, "WhatsApp group", message))
        self._log_action(agent_id, "post_in_whatsapp_group", message,
                         reached=len(targets))
        return f"{len(targets)} group members can read it."

    @tool(readonly=False)
    async def post_online(self, agent_id: int, platform: str,
                          message: str) -> str:
        """Post publicly on a social media platform you use.

        :param agent_id: Your own agent id.
        :param platform: One of x_twitter, tiktok, facebook, instagram, youtube, linkedin.
        :param message: The post text.
        :returns: Confirmation.
        """
        check_agent(agent_id)
        if platform not in ONLINE_PLATFORMS:
            raise ValueError(f"platform must be one of {ONLINE_PLATFORMS}")
        if platform not in self.world.channels[agent_id]:
            raise ValueError(f"you do not use {platform}")
        # Online reach is modelled in OASIS; here the post is only recorded.
        self._log_action(agent_id, "post_online", message, platform=platform)
        return "Posted."

    def _log_action(self, agent_id: int, action: str, text: str | None,
                    **extra) -> None:
        append_jsonl(self.log_path, {
            "type": "action", "date": self.t.date().isoformat(),
            "agent_id": agent_id, "action": action, "text": text, **extra})

    async def step(self, tick: int, t: datetime):
        # Messages sent this step are heard next step.
        self._delivering = dict(self._inbox)
        self._inbox = defaultdict(list)
        self.covered_until = t.date()
