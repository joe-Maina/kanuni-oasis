# Kanuni: AgentSociety 2 side of the policy simulator

Models the general public's offline life around a policy: how people hear
about it (radio, TV, WhatsApp, church, market talk, word of mouth), what it
does to their household budget (income, prices, rent and fees falling due,
loans), and what they do (talk, cut back, work more, borrow, protest,
contact their MP). AgentSociety 2 core code is not modified. Design notes:
`../../shared/DESIGN_agentsociety.md`.

## Files

| File | Role |
|---|---|
| `common.py` | Loads personas, livelihood profiles, channels, scenario; builds social ties; agent-id guard |
| `envs/information.py` | News by channel and word of mouth; logs every exposure |
| `envs/household.py` | Household ledger: income, obligations, prices and levers, loans |
| `envs/civic.py` | Protests, contacting the MP, community meetings |
| `agent.py` | `KanuniPersonAgent`: AS2's PersonAgent plus our observation and failure counters |
| `stance.py` | Strict daily stance survey (same question and parser as OASIS) |
| `run.py` | Our step loop (one agent's failure never stops the step) |
| `export.py` | Shared output, `household_days.jsonl`, `summary.json` |

Inputs live in `../../shared/`: `personas.json`, `countries/ke/`
(livelihood profiles, channels), `scenarios/` (bill timelines and levers).

## Run

```bash
# from agentic-society/, with the Ollama shim running on :8765
PYTHONPATH=. .venv/bin/python -m kanuni_as2.run --config kanuni_as2/configs/tiny_5.json
PYTHONPATH=. .venv/bin/python -m kanuni_as2.export kanuni_as2/runs/<run_id>
python3 ../shared/validation/score.py kanuni_as2/runs/<run_id>
```

## Run folder

| File | Contents |
|---|---|
| `output.jsonl` | Shared format, one row per agent per step |
| `household_days.jsonl` | One row per agent per simulated day: income, spending, missed payments, debt, protest |
| `summary.json` | Stance per step, failure counts, protest turnout by county, household summary |
| `agent_steps.jsonl` | Native record per agent-step: status, raw survey answer, model call counts, tools used |
| `env_log.jsonl` | Every exposure, action and household day as logged by the modules |
| `channels.json` | Each agent's channels and ties |
| `agents/` | AS2 agent workspaces (memory, profile) |

## Failure counting

`action_status`: `ok` (the model finished its day), `no_valid_finish` (AS2
hit the turn limit and wrote a fallback summary itself), `llm_error`,
`infra_error`. `interview_status`: `ok`, `parse_failed`, `no_valid_finish`,
`llm_error`, `infra_error`. AS2 also corrects some things on its own; these
are counted, not hidden: malformed model turns (it retries once within the
turn and then feeds back the error), and transport retries in its LLM
client (`summary.json` totals).

## Not leaking the outcome

Agents are told what people knew at the time (news events, public calls to
protest, protests other simulated agents joined), never where real protests
happened. That is what lets `shared/validation/score.py` compare protest
geography and timing with real records.
