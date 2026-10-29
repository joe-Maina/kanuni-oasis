"""Gateway settings, from KANUNI_GATEWAY_* environment variables (or a
.env file beside the service). Everything that ends up in a job script
is configured here, on the server: a request never supplies a command,
a script, or a path outside the root."""
from __future__ import annotations

from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class GatewaySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KANUNI_GATEWAY_", env_file=".env", extra="ignore")

    # Checkout of this repo, with both simulators' venvs built. Job
    # folders go under <root>/api_runs/<run id>.
    root: str
    # The Kanuni backend's SIM_GATEWAY_API_KEY; at least 32 characters.
    api_key: SecretStr

    # "process": each job is a background process in its own process
    # group on this machine. "slurm": sbatch/sacct/scancel.
    executor: Literal["process", "slurm"] = "process"
    max_request_mb: int = 50

    # Shell lines at the top of every job (module loads, env.sh).
    setup: str = ""
    oasis_python: str = "camel-oasis/oasis/.venv/bin/python"
    agentsociety_python: str = "agentic-society/.venv/bin/python"

    # The model the agents use; set in every submitted config.
    model_url: str = "http://127.0.0.1:8000/v1"
    vllm_serve_cmd: str = ""
    vllm_ready_timeout_s: int = 1800

    # Slurm only.
    slurm_partition: str | None = None
    slurm_gres: str = "gpu:1"
    slurm_time: str = "06:00:00"
    slurm_cpus: int = 8
    slurm_mem: str = "64G"
    slurm_extra: str = ""
