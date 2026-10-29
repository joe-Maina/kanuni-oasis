"""Jobs on this server: where their files live, what a submission may
contain, the job script, and the two ways of running one (a background
process, or Slurm).

The job script is built here from the gateway's own settings. A
submission brings only a run config (JSON) and input files (JSON), both
checked: the run id is a UUID, the simulator one of two, input files sit
under inputs/. Paths in the config (PATH_KEYS) are relative to the root
and resolved here, so the backend never needs to know where the root is;
any other absolute path is refused. It never brings a command.
"""
from __future__ import annotations

import io
import json
import os
import posixpath
import re
import shlex
import signal
import subprocess
import tarfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from kanuni_gateway.settings import GatewaySettings

SIMULATORS = ("oasis", "agentsociety")
RUN_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
INPUT_FILE = re.compile(r"^inputs/[a-z_]+\.json$")
# Config keys holding root-relative paths (a list for extra_timelines).
PATH_KEYS = ("personas", "events", "scenario", "outcomes", "profiles", "channels", "extra_timelines")

# Files sent back after a run, by simulator; personas.json is added from
# the config. AgentSociety's agents/ folder (per-agent memory) stays here.
RESULT_FILES = {
    "oasis": ("output.jsonl", "summary.json", "run_config.json"),
    "agentsociety": ("output.jsonl", "household_days.jsonl", "summary.json", "run_config.json"),
}

# Slurm-style states, as the backend groups them (cluster.py there).
COMPLETED, FAILED, CANCELLED, RUNNING = "COMPLETED", "FAILED", "CANCELLED", "RUNNING"
VANISHED, MISSING = "VANISHED", "MISSING"

_CMD_TIMEOUT_S = 60


class SubmissionError(ValueError):
    """The submission is refused; the message says why."""


class RunnerError(RuntimeError):
    """sbatch/sacct/scancel or starting a process failed."""


@dataclass(frozen=True)
class RunPaths:
    job_dir: Path      # config, job script, job log, inputs/
    results_dir: Path  # the folder the simulator writes

    @property
    def config(self) -> Path:
        return self.job_dir / "config.json"

    @property
    def script(self) -> Path:
        return self.job_dir / "job.sh"

    @property
    def log(self) -> Path:
        return self.job_dir / "job.log"

    @property
    def meta(self) -> Path:
        return self.job_dir / "job.json"


def run_paths(root: str, simulator: str, run_id: str) -> RunPaths:
    """The same layout the backend's SSH executors use."""
    base = Path(root)
    if simulator == "oasis":
        results = base / "camel-oasis/oasis/kanuni/runs" / run_id
    else:
        results = base / "agentic-society/kanuni_as2/runs" / run_id
    return RunPaths(job_dir=base / "api_runs" / run_id, results_dir=results)


def check_run_id(run_id: str) -> None:
    if not RUN_ID.match(run_id):
        raise SubmissionError("run_id must be a lowercase UUID")


def check_submission(settings: GatewaySettings, run_id: str, simulator: str, config: dict,
                     files: dict[str, str]) -> dict:
    """The config to write, with its paths made absolute under the root
    and the gateway's model URL, or SubmissionError."""
    check_run_id(run_id)
    if simulator not in SIMULATORS:
        raise SubmissionError(f"simulator must be one of {', '.join(SIMULATORS)}")
    if config.get("run_id") != run_id:
        raise SubmissionError("config.run_id must equal run_id")
    for name, text in files.items():
        if not INPUT_FILE.match(name):
            raise SubmissionError(f"input file {name!r} must be named inputs/<name>.json")
        try:
            json.loads(text)
        except ValueError as exc:
            raise SubmissionError(f"input file {name} is not JSON: {exc}") from exc

    def walk(value, where: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{where}.{key}")
        elif isinstance(value, list):
            for i, item in enumerate(value):
                walk(item, f"{where}[{i}]")
        elif isinstance(value, str) and value.startswith("/"):
            raise SubmissionError(f"config{where}: paths are relative to the gateway root")
    walk(config, "")

    def resolve(value, key: str) -> str:
        if not isinstance(value, str) or not value or ".." in value.split("/"):
            raise SubmissionError(f"config.{key} must be a path inside the gateway root")
        return posixpath.join(settings.root, value)

    config = json.loads(json.dumps(config))
    for key in PATH_KEYS:
        if key in config:
            value = config[key]
            config[key] = [resolve(v, key) for v in value] if isinstance(value, list) else resolve(value, key)
    if isinstance(config.get("model"), dict):
        config["model"]["url"] = settings.model_url
    return config


def build_job_script(settings: GatewaySettings, simulator: str, run_id: str, paths: RunPaths) -> str:
    """The job script for one run. #SBATCH lines are comments to bash, so
    the same script serves both executors."""
    root = settings.root
    q = shlex.quote
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=kanuni-{run_id[:12]}",
        f"#SBATCH --output={paths.log}",
        f"#SBATCH --gres={settings.slurm_gres}",
        f"#SBATCH --time={settings.slurm_time}",
        f"#SBATCH --cpus-per-task={settings.slurm_cpus}",
        f"#SBATCH --mem={settings.slurm_mem}",
    ]
    if settings.slurm_partition:
        lines.append(f"#SBATCH --partition={settings.slurm_partition}")
    for extra in settings.slurm_extra.splitlines():
        if extra.strip():
            lines.append(f"#SBATCH {extra.strip()}")
    lines += ["set -euo pipefail", f"cd {q(root)}"]
    if settings.setup.strip():
        lines += ["# site setup", settings.setup.strip()]
    if settings.vllm_serve_cmd.strip():
        models = settings.model_url.rstrip("/") + "/models"
        lines += [
            "# start the model server for this job and wait until it answers",
            f"( {settings.vllm_serve_cmd.strip()} ) > {q(str(paths.job_dir / 'model_server.log'))} 2>&1 &",
            "MODEL_PID=$!",
            "trap 'kill $MODEL_PID 2>/dev/null || true' EXIT",
            f"for i in $(seq 1 {max(1, settings.vllm_ready_timeout_s // 10)}); do",
            f"  curl -sf {q(models)} > /dev/null && break",
            "  kill -0 $MODEL_PID 2>/dev/null || { echo 'model server exited before it was ready'; exit 3; }",
            "  sleep 10",
            "done",
            f"curl -sf {q(models)} > /dev/null || {{ echo 'model server not ready in time'; exit 3; }}",
        ]
    if simulator == "oasis":
        py = settings.oasis_python
        py = py if py.startswith("/") else posixpath.join(root, py)
        lines += [
            f"cd {q(posixpath.join(root, 'camel-oasis/oasis'))}",
            f"{q(py)} -m kanuni.run_oasis --config {q(str(paths.config))} --run-id {q(run_id)}",
        ]
    else:
        py = settings.agentsociety_python
        py = py if py.startswith("/") else posixpath.join(root, py)
        lines += [
            f"cd {q(posixpath.join(root, 'agentic-society'))}",
            f"PYTHONPATH=. {q(py)} -m kanuni_as2.run --config {q(str(paths.config))} --run-id {q(run_id)}",
        ]
    return "\n".join(lines) + "\n"


class Runner(Protocol):
    def submit(self, paths: RunPaths) -> str: ...
    def state(self, paths: RunPaths, native_id: str) -> str: ...
    def cancel(self, paths: RunPaths, native_id: str) -> None: ...


def _run(argv: list[str]) -> str:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=_CMD_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RunnerError(f"{argv[0]}: {exc}") from exc
    if proc.returncode != 0:
        raise RunnerError((proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()[-1000:])
    return proc.stdout


class SlurmRunner:
    def submit(self, paths: RunPaths) -> str:
        """sbatch --parsable prints '<job id>' or '<job id>;<cluster>'."""
        out = _run(["sbatch", "--parsable", str(paths.script)]).strip()
        job_id = out.split(";")[0].strip()
        if not job_id.isdigit():
            raise RunnerError(f"unexpected sbatch output: {out!r}")
        return job_id

    def state(self, paths: RunPaths, native_id: str) -> str:
        """sacct knows finished jobs; one submitted seconds ago may only be
        in squeue yet."""
        out = _run(["sacct", "-j", native_id, "-n", "-X", "-P", "-o", "State"]).strip()
        if not out:
            out = _run(["squeue", "-h", "-j", native_id, "-o", "%T"]).strip()
        return out.splitlines()[0] if out else "UNKNOWN"

    def cancel(self, paths: RunPaths, native_id: str) -> None:
        _run(["scancel", native_id])


class ProcessRunner:
    """The job script as a background process in its own process group,
    tracked by files in its job folder (as the backend's SshCluster does):

        job.pid        the process group id
        job.exit       the script's exit code, written when it ends
        job.cancelled  written by cancel(), before the group is killed

    A job outlives a gateway restart; its state is read from the files.
    """

    def submit(self, paths: RunPaths) -> str:
        d = paths.job_dir
        for name in ("job.exit", "job.cancelled", "job.pid"):
            (d / name).unlink(missing_ok=True)
        inner = "bash job.sh > job.log 2>&1; echo $? > job.exit"
        try:
            proc = subprocess.Popen(["bash", "-c", inner, f"kanuni-job {d}"], cwd=d,
                                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, start_new_session=True)
        except OSError as exc:
            raise RunnerError(f"could not start the job: {exc}") from exc
        # Reap it when it ends, so it doesn't linger as a zombie.
        threading.Thread(target=proc.wait, daemon=True).start()
        (d / "job.pid").write_text(str(proc.pid))
        return str(proc.pid)

    def state(self, paths: RunPaths, native_id: str) -> str:
        d = paths.job_dir
        if (d / "job.cancelled").exists():
            return CANCELLED
        if (d / "job.exit").exists():
            return COMPLETED if (d / "job.exit").read_text().strip() == "0" else FAILED
        if _is_job_process(native_id, d):
            return RUNNING
        # The exit file is written as the script ends; give it a moment.
        time.sleep(0.2)
        if (d / "job.exit").exists():
            return self.state(paths, native_id)
        return VANISHED

    def cancel(self, paths: RunPaths, native_id: str) -> None:
        (paths.job_dir / "job.cancelled").touch()
        try:
            os.killpg(int(native_id), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, ValueError):
            pass


def _is_job_process(pid: str, job_dir: Path) -> bool:
    """The pid is alive and is this job's (its command line names the job
    folder), so a reused pid isn't mistaken for the job."""
    if not pid.isdigit():
        return False
    try:
        out = subprocess.run(["ps", "-ww", "-o", "args=", "-p", pid], capture_output=True, text=True,
                             timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return False
    return str(job_dir) in out and "<defunct>" not in out


def make_runner(settings: GatewaySettings) -> Runner:
    return SlurmRunner() if settings.executor == "slurm" else ProcessRunner()


def tail(path: Path, lines: int) -> str:
    if not path.exists():
        return ""
    with path.open(errors="replace") as f:
        return "".join(deque(f, maxlen=lines))


def results_archive(paths: RunPaths, simulator: str, out: io.BufferedIOBase) -> None:
    """Writes a run's result files, and its personas as personas.json, to
    out as a gzip tar with flat names. A missing file raises
    FileNotFoundError, so a run whose output never got written is
    reported, not sent half-empty."""
    members = [(name, paths.results_dir / name) for name in RESULT_FILES[simulator]]
    personas = json.loads(paths.config.read_text()).get("personas")
    if personas:
        members.append(("personas.json", Path(personas)))
    for name, path in members:
        if not path.is_file():
            raise FileNotFoundError(f"{name} was not written")
    with tarfile.open(fileobj=out, mode="w:gz") as tar:
        for name, path in members:
            tar.add(path, arcname=name, recursive=False)
