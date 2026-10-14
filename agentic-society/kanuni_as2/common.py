"""Shared helpers for the Kanuni AgentSociety modules.

Everything here is deterministic given the run seed, so two runs with the
same seed see the same channels, ties and random draws.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import random
from datetime import date
from pathlib import Path
from typing import Any

# Id of the agent whose turn is running. Set by KanuniPersonAgent around
# every environment request; env tools check it so the router model cannot
# act for a different agent by passing the wrong agent_id.
CURRENT_AGENT: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "kanuni_current_agent", default=None)

# Position inside a profile's [low, high] range for each income band.
BAND_POSITION = {"very_low": 0.1, "low": 0.3, "middle": 0.5,
                 "upper_middle": 0.7, "high": 0.9, "unknown": 0.5,
                 "dependent": 0.0}


class WrongAgent(Exception):
    """An env tool was called with another agent's id."""


def check_agent(agent_id: int) -> None:
    current = CURRENT_AGENT.get()
    if current is not None and int(agent_id) != current:
        raise WrongAgent(
            f"agent {current} tried to act as agent {agent_id}; "
            f"use agent_id={current}")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path: str | Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def value(field: Any) -> Any:
    """Unwrap {"value": x, "src"/"assumed": ...} entries."""
    return field["value"] if isinstance(field, dict) and "value" in field \
        else field


def in_range(rng_or_pair: Any, position: float) -> float:
    lo, hi = rng_or_pair if isinstance(rng_or_pair, list) else (
        rng_or_pair, rng_or_pair)
    return lo + (hi - lo) * position


def stable_rng(*keys: Any) -> random.Random:
    """A Random seeded from the given keys (stable across processes)."""
    digest = hashlib.sha256("|".join(map(str, keys)).encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def chance(p: float, *keys: Any) -> bool:
    return stable_rng(*keys).random() < p


class World:
    """Personas, livelihood profiles, channels, ties and the scenario."""

    def __init__(self, personas_path: str, profiles_path: str,
                 channels_path: str, scenario_path: str, seed: int,
                 agent_ids: list[int] | None = None):
        self.seed = seed
        personas = load_json(personas_path)
        if agent_ids is not None:
            keep = set(agent_ids)
            personas = [p for p in personas if p["agent_id"] in keep]
        self.personas: dict[int, dict] = {p["agent_id"]: p for p in personas}
        self.econ = load_json(profiles_path)
        self.profile_of_livelihood = {
            l: pr for pr in self.econ["profiles"] for l in pr["livelihoods"]}
        self.channels_cfg = load_json(channels_path)
        scenario_path = Path(scenario_path)
        self.scenario = load_json(scenario_path)
        timeline = load_json(scenario_path.parent /
                             self.scenario["events_file"])
        events = list(timeline["events"])
        # Who posts each event: news desks, or accounts from optional layers
        # (activists, influencers, pro-government), with their kind.
        self.accounts = {a["account_id"]: {**a, "kind": a.get("kind", "news")}
                         for a in timeline["media_accounts"]}
        for extra in self.scenario.get("extra_events_files", []):
            layer = load_json(scenario_path.parent / extra)
            events += layer["events"]
            self.accounts.update({a["account_id"]: a
                                  for a in layer["media_accounts"]})
        dropped = set(self.scenario.get("drop_events", []))
        self.events = sorted((e for e in events
                              if e["event_id"] not in dropped),
                             key=lambda e: e["date"])
        self.channels = {aid: self._channels_for(p)
                         for aid, p in self.personas.items()}
        self.ties = self._build_ties()

    def profile(self, agent_id: int) -> dict:
        return self.profile_of_livelihood[self.personas[agent_id]["livelihood"]]

    def band_position(self, agent_id: int) -> float:
        p = self.personas[agent_id]
        jitter = stable_rng(self.seed, "band", agent_id).uniform(-0.1, 0.1)
        return min(1.0, max(0.0,
                            BAND_POSITION.get(p["income_band"], 0.5) + jitter))

    def _channels_for(self, p: dict) -> list[str]:
        known = self.channels_cfg["channels"]
        chans = [h for h in p.get("media_habits", []) if h in known]
        attrs = {**p, "profile_id": self.profile_of_livelihood[
            p["livelihood"]]["profile_id"]}
        for rule in self.channels_cfg["offline_rules"]:
            ok = True
            for key, want in rule["if"].items():
                have = attrs.get(key)
                ok &= have in want if isinstance(want, list) else have == want
            if ok and rule["grant"] not in chans:
                chans.append(rule["grant"])
        return chans

    def _build_ties(self) -> dict[int, dict[str, list[int]]]:
        cfg = self.channels_cfg["word_of_mouth"]
        ids = sorted(self.personas)
        ties: dict[int, dict[str, list[int]]] = {}
        for aid in ids:
            p = self.personas[aid]
            rng = stable_rng(self.seed, "ties", aid)
            prof = self.profile(aid)["profile_id"]
            same_county = [o for o in ids if o != aid and
                           self.personas[o]["county"] == p["county"]]
            workmates = [o for o in same_county
                         if self.profile(o)["profile_id"] == prof]
            others = [o for o in ids if o != aid]
            ties[aid] = {
                "neighbours": rng.sample(same_county, min(
                    cfg["neighbours_per_agent"], len(same_county))),
                "workmates": rng.sample(workmates, min(
                    cfg["workmates_per_agent"], len(workmates))),
                "family": rng.sample(others, min(
                    cfg["family_per_agent"], len(others))),
            }
        return ties

    def events_between(self, after: date | None, upto: date) -> list[dict]:
        """Events dated after `after` (None: from the start) up to `upto`."""
        return [e for e in self.events
                if (after is None or date.fromisoformat(e["date"]) > after)
                and date.fromisoformat(e["date"]) <= upto]

    def protest_call(self, day: date) -> dict | None:
        """The public call to protest on `day`, if any (same for everyone)."""
        for c in self.scenario.get("protest_calls", []):
            if date.fromisoformat(c["date"]) == day:
                return c
        return None
