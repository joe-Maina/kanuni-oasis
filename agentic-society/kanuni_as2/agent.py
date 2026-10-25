"""KanuniPersonAgent: AS2's PersonAgent plus failure counting and our observation.

Subclassing keeps AS2's core untouched. Changes:
  * The daily observation ("<observe>") is built by our modules in code
    (news through the person's channels, household money, protests today)
    instead of asking the env router's model. Exposure is then exact and
    costs no model call.
  * Every env request runs with CURRENT_AGENT set, so env tools can refuse a
    request that names another agent's id.
  * Counters for what AS2 corrects on its own: malformed model turns
    (including AS2's within-turn retry) and loops that ended without the
    model finishing (AS2 then synthesizes a fallback). The runner reads and
    resets them after each call, so they can be reported, not hidden.
"""
from __future__ import annotations

from typing import Any

from agentsociety2.agent import PersonAgent

from kanuni_as2.common import CURRENT_AGENT


class KanuniPersonAgent(PersonAgent):

    def reset_counters(self) -> None:
        self.kanuni_counts = {"llm_calls": 0, "malformed_turns": 0,
                              "fallback": False, "env_requests": 0,
                              "tools": []}
        self.kanuni_errors: list[str] = []

    def _modules(self) -> dict[str, Any]:
        env = self._env
        mods = getattr(env, "env_modules", None)
        if mods is None:
            raise RuntimeError("KanuniPersonAgent needs an in-process router")
        return {type(m).__name__: m for m in mods}

    def observation_text(self) -> str:
        # Information, household, then civic life or the policy's actions
        # (KanuniCivicEnv or KanuniPolicyEnv), in the router's order.
        parts = [m.observe_for(self.id) for m in self._modules().values()
                 if hasattr(m, "observe_for")]
        return "\n\n".join(p for p in parts if p)

    async def ask_env(self, ctx: dict, message: str, readonly: bool,
                      template_mode: bool = False, trace_id: str | None = None,
                      parent_span_id: str | None = None):
        if message.strip() == "<observe>":
            return ctx, self.observation_text()
        self.kanuni_counts["env_requests"] += 1
        token = CURRENT_AGENT.set(self.id)
        try:
            return await super().ask_env(
                ctx, message, readonly, template_mode=template_mode,
                trace_id=trace_id, parent_span_id=parent_span_id)
        finally:
            CURRENT_AGENT.reset(token)

    async def _complete_react_once(self, messages, *, readonly):
        self.kanuni_counts["llm_calls"] += 1
        return await super()._complete_react_once(messages, readonly=readonly)

    def _parse_react_turn(self, response, *, readonly):
        turn = super()._parse_react_turn(response, readonly=readonly)
        if turn.error:
            self.kanuni_counts["malformed_turns"] += 1
            self.kanuni_errors.append(str(turn.error)[:300])
        return turn

    async def _execute_react_tools(self, decisions, *, readonly):
        self.kanuni_counts["tools"].extend(d.action for d in decisions)
        return await super()._execute_react_tools(decisions, readonly=readonly)

    def _synthesize_ask_fallback(self) -> str:
        self.kanuni_counts["fallback"] = True
        return super()._synthesize_ask_fallback()

    def _synthesize_step_fallback_memory(self, *args, **kwargs):
        self.kanuni_counts["fallback"] = True
        return super()._synthesize_step_fallback_memory(*args, **kwargs)
