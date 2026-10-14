# Kanuni: OASIS side of the policy simulator

Our code for running the Finance Bill 2024 backtest on OASIS. OASIS core code
is not modified. Shared, framework-neutral inputs live in
`../../../shared/` (`personas.json`, `events.json`) and are also used by the
AgentSociety prototype.

## Setup

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e .   # from the repo root
```

Create `kanuni/.env` (git-ignored) with the model endpoint:

```
KANUNI_OLLAMA_BASE=https://<host>
KANUNI_OLLAMA_TOKEN=<token>
KANUNI_OLLAMA_THINK=false
```

## Run

```bash
# 1. OpenAI-compatible shim in front of Ollama's /api/chat (the proxy does
#    not expose /v1, which CAMEL needs). Keep it running.
.venv/bin/python -m kanuni.ollama_shim --port 8765

# 2. Check the model can make tool calls through CAMEL.
.venv/bin/python -m kanuni.check_model

# 3. Simulation (writes kanuni/runs/<run_id>/, then exports).
.venv/bin/python -m kanuni.run_oasis --config kanuni/configs/smoke_100.json

# Re-export an existing run.
.venv/bin/python -m kanuni.export kanuni/runs/<run_id>
```

`--run-id` overrides the config's `run_id` (a run never overwrites an
existing folder).

### On the VM

Runs go on the Ollama VM (`ssh kanuny-vm`) so home internet drops cannot
break them. Everything lives on the data volume, `/mnt/volume_qko3nkn/kanuni`
(the root disk is nearly full); `source /mnt/volume_qko3nkn/kanuni/env.sh`
first, which points caches and temp files there and enters the repo. There
`kanuni/.env` points the shim at `http://localhost:11434`, so no internet is
involved. The shim and runs live in the tmux session `kanuni`:

```bash
ssh -t kanuny-vm tmux attach -t kanuni   # Ctrl-b n: next window; Ctrl-b d: detach
rsync -az kanuny-vm:/mnt/volume_qko3nkn/kanuni/camel-oasis/oasis/kanuni/runs/<run_id> kanuni/runs/
```

`upstream_example_10.py` is OASIS's own vLLM example cut down to 10 agents
and pointed at our model (smoke-test step 1).

## What a run does

One tick is one simulated day, from `start_date` to `end_date`. Each day:
media accounts post that day's events (day 0 also gets every earlier event);
the recommendation table is rebuilt; each persona agent acts once via the
LLM; then each is asked a private stance/intent question (`stance.py`) that
is not written to its memory. Only personas whose `connectivity` is in
`connectivity_filter` take part (`heavy_online` for now).

## Run folder

| File | Contents |
|---|---|
| `output.jsonl` | Shared output format, one row per agent per day |
| `summary.json` | Per-day stance summary and failure counts |
| `agent_days.jsonl` | Native per-agent-day record: tool calls, raw interview answer, errors, timings |
| `events_injected.jsonl` | Which event became which post, on which day |
| `oasis.db` | OASIS SQLite database (`trace` table has every action) |
| `id_map.json` | OASIS user id to persona `agent_id` or media account |
| `log/` | OASIS's own logs |

## Failure counting

Model output is never retried or repaired. The only retry is in the shim,
for connection failures where the request never reached the server.
`action_status` per agent-day is `ok`, `no_tool_call` (text reply only),
`llm_error`, `action_rejected` (every tool call refused by the platform),
`infra_error` (network, timeout or upstream failure, not the model's fault),
or `context_overflow` (the shim refused a prompt too long for the context
window).
Interviews must be a JSON object with an integer stance from -2 to 2 and an
intent in `none|post|attend_protest|contact_mp`; anything else is
`parse_failed`. `parse_failed` in the shared output is true if the agent-day
has no valid action or stance for any reason; `summary.json` reports model
failures and infra failures separately.

## Keeping prompts bounded

Without limits an agent's prompt grows every day: CAMEL keeps every day's
full feed in memory (it does not know this model's context size, so it
never trims), and OASIS shows every comment on every post. In
`smoke_100_seed0_vm` prompts reached 40k+ tokens by day 3; Ollama silently
cut them to the last ~8k, dropping the persona, and the model stopped
choosing actions. Settings now used (recorded in each run's config):

- `platform.following_post_count` 3 and `refresh_rec_post_count` 2;
- `max_comments_per_post` 3: our feed subclass keeps the most-liked
  comments (the platform trace still records the full feed);
- `memory_window_messages` 8: the persona plus the last 8 messages
  (about two days). CAMEL cannot trim by tokens here without summarising
  through extra model calls;
- shim: 32k context (`KANUNI_OLLAMA_NUM_CTX`), and prompts estimated not to
  fit are refused, recorded as `context_overflow`, instead of being cut.

## Known limits

- The feed shows followed accounts' posts ranked by likes, not recency
  (`following_post_count`). With few events this is fine; with many, new
  events can be crowded out. Check `saw_any_event` in `summary.json`.
- Day -1 in the database is setup (sign-up, following media accounts).
- `qwen3.5:27b` was trained after June 2024 and knows the outcome. Reportable
  runs need a model with an earlier cutoff.
