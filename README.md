# Kanuni policy simulator (prototype)

LLM agents with Kenyan personas react to policies, simulated in two
frameworks:

- **OASIS** (online public): agents on a simulated X/Twitter.
  Our code: `camel-oasis/oasis/kanuni/`.
- **AgentSociety 2** (general public, offline life): news by radio,
  WhatsApp, church and word of mouth; household budgets with income,
  obligations and debt; protests and contacting MPs.
  Our code: `agentic-society/kanuni_as2/`.

Both write the same per-agent, per-day output format, so results can be
compared. Neither framework's core code is modified.

First case: the Kenya Finance Bill 2024 backtest. Next: the Artificial
Intelligence Bill 2026.

## Layout

| Path | Contents |
|---|---|
| `shared/` | Framework-neutral inputs and docs: personas, event timeline, activist layer, scenarios, Kenya livelihood profiles and channels, validation data and scorer, design notes, experiment log |
| `camel-oasis/oasis/kanuni/` | OASIS runner, Ollama shim, stance survey, exporter, configs |
| `camel-oasis/oasis/personas/` | Hand-written persona source file |
| `agentic-society/kanuni_as2/` | AgentSociety 2 env modules, agent subclass, runner, exporter, configs |
| `setup.sh` | Fetches OASIS and AgentSociety 2 at pinned commits and builds the venvs |

## Setup

```bash
./setup.sh            # needs Python 3.11 and curl
```

Then see `camel-oasis/oasis/kanuni/README.md` and
`agentic-society/kanuni_as2/README.md` for running, and
`shared/DESIGN_agentsociety.md` and `shared/EXPERIMENTS.md` for design and
results.

## Upstream

- OASIS, camel-ai/oasis @ `dded86a`, Apache-2.0
- AgentSociety 2, tsinghua-fib-lab/AgentSociety @ `8cc5bb9` (2.9.3), Apache-2.0

Results so far use `qwen3.5:27b`, which was trained after June 2024 and
knows how the Finance Bill played out. They test the pipeline and are not
reportable; reportable backtests need a model with an earlier training
cutoff.
