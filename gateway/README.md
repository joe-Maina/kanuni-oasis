# Simulation gateway

An HTTPS service on the GPU server. The Kanuni backend uses it (`SIM_EXECUTOR=http`) to submit runs, follow them and collect their results, so it no longer needs SSH.

| Endpoint | What it does |
|---|---|
| `POST /v1/jobs` | Submit a run: `{run_id, simulator, config, files}`. Idempotent per run id. |
| `GET /v1/jobs/{run_id}` | State, using Slurm-style names (`RUNNING`, `COMPLETED`, `FAILED`, `CANCELLED`, `VANISHED`, `MISSING`). |
| `POST /v1/jobs/{run_id}/cancel` | Stop the whole process group, or `scancel`. |
| `GET /v1/jobs/{run_id}/log?lines=60` | The tail of `job.log`. |
| `GET /v1/jobs/{run_id}/results` | Result files plus `personas.json` as a `.tar.gz`, with an `X-Content-SHA256` header. |
| `GET /healthz` | Liveness check. No authentication. |

Every `/v1` request carries `Authorization: Bearer <API key>`; keep it behind HTTPS.

The gateway builds each job script from its own settings. A request supplies only a JSON config and JSON input files (`inputs/<name>.json`), and these rules apply:

- The run id must be a UUID.
- The simulator must be `oasis` or `agentsociety`.
- Paths in the config (`personas`, `events`, `scenario`, …) are relative to `KANUNI_GATEWAY_ROOT`; the gateway makes them absolute, and refuses any other absolute path or `..`.
- The model URL is set to `KANUNI_GATEWAY_MODEL_URL`.

## Settings (environment, or `.env` here)

```
KANUNI_GATEWAY_ROOT=/path/to/kanuny-policy-sim
KANUNI_GATEWAY_API_KEY=...                       # openssl rand -hex 32; = backend SIM_GATEWAY_API_KEY
KANUNI_GATEWAY_EXECUTOR=process                  # or slurm
KANUNI_GATEWAY_MODEL_URL=http://127.0.0.1:8000/v1
KANUNI_GATEWAY_SETUP=source /path/to/env.sh      # optional shell lines at the top of each job
```

See `kanuni_gateway/settings.py` for the rest (python paths, vLLM start command, Slurm resources).

## Run

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn kanuni_gateway.app:app --host 127.0.0.1 --port 8700
```

Bind it to localhost and put a TLS proxy such as Caddy in front of it. Allow only the backend's address through the firewall.

## Test

```
.venv/bin/python -m pytest -q tests
```
