"""The gateway against a fake runner (submit/state/cancel recorded, no
processes), plus ProcessRunner on one real two-line job."""
from __future__ import annotations

import hashlib
import io
import json
import tarfile
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kanuni_gateway import jobs
from kanuni_gateway.app import create_app
from kanuni_gateway.settings import GatewaySettings

KEY = "k" * 40
RUN = "0b6f1c2e-1d2a-4c3b-9e8f-7a6b5c4d3e2f"


class FakeRunner:
    def __init__(self):
        self.submitted, self.cancelled, self.states = [], [], {}

    def submit(self, paths):
        self.submitted.append(paths)
        return "4242"

    def state(self, paths, native_id):
        return self.states.get(native_id, "RUNNING")

    def cancel(self, paths, native_id):
        self.cancelled.append(native_id)


@pytest.fixture
def gateway(tmp_path):
    settings = GatewaySettings(root=str(tmp_path), api_key=KEY, model_url="http://127.0.0.1:8000/v1",
                               setup="source env.sh")
    runner = FakeRunner()
    return TestClient(create_app(settings, runner)), runner, tmp_path


def call(client, method, path, body=None, *, key=KEY):
    return client.request(method, path, json=body, headers={"Authorization": f"Bearer {key}"})


def submission(root, **overrides):
    inputs = f"api_runs/{RUN}/inputs"
    config = {"run_id": RUN, "personas": f"{inputs}/personas.json", "events": f"{inputs}/events.json",
              "model": {"name": "qwen", "url": "http://elsewhere:9000/v1"}, "extra_timelines": ["shared/a.json"]}
    return {"run_id": RUN, "simulator": "oasis", "config": config,
            "files": {"inputs/personas.json": "[]", "inputs/events.json": "[]"}, **overrides}


def test_healthz_needs_no_signature(gateway):
    client, _, _ = gateway
    assert client.get("/healthz").json() == {"ok": True}


def test_requests_need_the_api_key(gateway):
    client, runner, root = gateway
    assert call(client, "GET", f"/v1/jobs/{RUN}", key="x" * 40).status_code == 401
    assert client.get(f"/v1/jobs/{RUN}").status_code == 401
    assert client.get(f"/v1/jobs/{RUN}", headers={"Authorization": KEY}).status_code == 401
    assert client.post("/v1/jobs", json=submission(root)).status_code == 401 and not runner.submitted


def test_submit_writes_inputs_config_and_script_then_starts_once(gateway):
    client, runner, root = gateway
    r = call(client, "POST", "/v1/jobs", submission(root))
    assert r.status_code == 200 and r.json() == {"job_id": RUN, "state": "PENDING", "created": True}
    job_dir = root / "api_runs" / RUN
    assert json.loads((job_dir / "inputs/personas.json").read_text()) == []
    config = json.loads((job_dir / "config.json").read_text())
    assert config["model"]["url"] == "http://127.0.0.1:8000/v1"  # the gateway's model, not the request's
    # Root-relative paths become absolute on this server.
    assert config["personas"] == f"{root}/api_runs/{RUN}/inputs/personas.json"
    assert config["extra_timelines"] == [f"{root}/shared/a.json"]
    script = (job_dir / "job.sh").read_text()
    assert "source env.sh" in script and f"-m kanuni.run_oasis --config {job_dir}/config.json" in script
    assert json.loads((job_dir / "job.json").read_text())["native_id"] == "4242"
    # A retried submission is the same job.
    again = call(client, "POST", "/v1/jobs", submission(root))
    assert again.json()["created"] is False and len(runner.submitted) == 1


@pytest.mark.parametrize("change, message", [
    ({"run_id": "../../etc"}, "UUID"),
    ({"simulator": "bash"}, "simulator"),
    ({"files": {"job.sh": "{}"}}, "inputs/<name>.json"),
    ({"files": {"inputs/../job.sh": "{}"}}, "inputs/<name>.json"),
    ({"files": {"inputs/personas.json": "not json"}}, "not JSON"),
])
def test_submissions_outside_the_rules_are_refused(gateway, change, message):
    client, runner, root = gateway
    r = call(client, "POST", "/v1/jobs", submission(root, **change))
    assert r.status_code == 422 and message in r.json()["detail"] and not runner.submitted


@pytest.mark.parametrize("key, value, message", [
    ("personas", "/etc/passwd", "relative to the gateway root"),
    ("extra_timelines", ["../elsewhere/a.json"], "inside the gateway root"),
    ("events", "shared/../../x.json", "inside the gateway root"),
    ("events", "", "inside the gateway root"),
    ("notes", "/home/someone", "relative to the gateway root"),
])
def test_config_paths_must_stay_under_the_root(gateway, key, value, message):
    client, runner, root = gateway
    body = submission(root)
    body["config"][key] = value
    r = call(client, "POST", "/v1/jobs", body)
    assert r.status_code == 422 and message in r.json()["detail"] and not runner.submitted


def test_config_must_be_for_the_same_run(gateway):
    client, _, root = gateway
    body = submission(root)
    body["config"]["run_id"] = "another"
    assert call(client, "POST", "/v1/jobs", body).status_code == 422


def test_state_cancel_and_log(gateway):
    client, runner, root = gateway
    assert call(client, "GET", f"/v1/jobs/{RUN}").json()["state"] == "MISSING"
    call(client, "POST", "/v1/jobs", submission(root))
    runner.states["4242"] = "COMPLETED"
    assert call(client, "GET", f"/v1/jobs/{RUN}").json() == {"job_id": RUN, "state": "COMPLETED"}
    (root / "api_runs" / RUN / "job.log").write_text("".join(f"line {i}\n" for i in range(100)))
    assert call(client, "GET", f"/v1/jobs/{RUN}/log?lines=2").text == "line 98\nline 99\n"
    assert call(client, "POST", f"/v1/jobs/{RUN}/cancel").status_code == 200 and runner.cancelled == ["4242"]


def test_unknown_jobs_cannot_be_cancelled_or_read(gateway):
    client, _, _ = gateway
    assert call(client, "POST", f"/v1/jobs/{RUN}/cancel").status_code == 404
    assert call(client, "GET", f"/v1/jobs/{RUN}/results").status_code == 404


def test_results_come_back_as_a_checksummed_archive_with_personas(gateway):
    client, _, root = gateway
    call(client, "POST", "/v1/jobs", submission(root))
    out = root / "camel-oasis/oasis/kanuni/runs" / RUN
    out.mkdir(parents=True)
    for name in jobs.RESULT_FILES["oasis"]:
        (out / name).write_text(f"{name}\n")
    r = call(client, "GET", f"/v1/jobs/{RUN}/results")
    assert r.status_code == 200
    assert hashlib.sha256(r.content).hexdigest() == r.headers["X-Content-SHA256"]
    with tarfile.open(fileobj=io.BytesIO(r.content)) as tar:
        names = tar.getnames()
        assert tar.extractfile("output.jsonl").read() == b"output.jsonl\n"
    assert sorted(names) == sorted([*jobs.RESULT_FILES["oasis"], "personas.json"])


def test_results_missing_a_file_are_refused(gateway):
    client, _, root = gateway
    call(client, "POST", "/v1/jobs", submission(root))
    r = call(client, "GET", f"/v1/jobs/{RUN}/results")
    assert r.status_code == 409 and "output.jsonl was not written" in r.json()["detail"]


def test_a_short_secret_is_refused(tmp_path):
    with pytest.raises(ValueError):
        create_app(GatewaySettings(root=str(tmp_path), api_key="short"), FakeRunner())


def _wait(runner, paths, native_id, done=("COMPLETED", "FAILED", "CANCELLED", "VANISHED")):
    for _ in range(100):
        state = runner.state(paths, native_id)
        if state in done:
            return state
        time.sleep(0.05)
    return state


def test_process_runner_runs_a_job_and_reads_its_exit(tmp_path):
    paths = jobs.run_paths(str(tmp_path), "oasis", RUN)
    paths.job_dir.mkdir(parents=True)
    paths.script.write_text("echo hello\n")
    runner = jobs.ProcessRunner()
    assert _wait(runner, paths, runner.submit(paths)) == "COMPLETED"
    assert paths.log.read_text() == "hello\n"
    paths.script.write_text("exit 4\n")
    assert _wait(runner, paths, runner.submit(paths)) == "FAILED"


def test_process_runner_cancels_the_whole_job(tmp_path):
    paths = jobs.run_paths(str(tmp_path), "oasis", RUN)
    paths.job_dir.mkdir(parents=True)
    paths.script.write_text("sleep 30\n")
    runner = jobs.ProcessRunner()
    native_id = runner.submit(paths)
    assert _wait(runner, paths, native_id, done=("RUNNING",)) == "RUNNING"
    runner.cancel(paths, native_id)
    assert runner.state(paths, native_id) == "CANCELLED"


def test_job_script_for_slurm_with_partition_and_extras(tmp_path):
    settings = GatewaySettings(root=str(tmp_path), api_key=KEY, slurm_partition="gpu",
                               setup="module load cuda/12.8", slurm_extra="--account=kanuni\n--qos=normal")
    paths = jobs.run_paths(settings.root, "oasis", RUN)
    script = jobs.build_job_script(settings, "oasis", RUN, paths)
    assert "#SBATCH --partition=gpu" in script
    assert "#SBATCH --account=kanuni" in script and "#SBATCH --qos=normal" in script
    assert "module load cuda/12.8" in script and f"--run-id {RUN}" in script
    assert str(paths.results_dir).endswith(f"camel-oasis/oasis/kanuni/runs/{RUN}")
    assert "MODEL_PID" not in script  # no model server unless configured


def test_job_script_starts_and_waits_for_model_server(tmp_path):
    settings = GatewaySettings(root=str(tmp_path), api_key=KEY, vllm_serve_cmd="bash vllm/serve.sh",
                               vllm_ready_timeout_s=600)
    paths = jobs.run_paths(settings.root, "agentsociety", RUN)
    script = jobs.build_job_script(settings, "agentsociety", RUN, paths)
    assert "bash vllm/serve.sh" in script and "trap 'kill $MODEL_PID" in script
    assert "seq 1 60" in script and "curl -sf http://127.0.0.1:8000/v1/models" in script
    assert "PYTHONPATH=. " in script and "-m kanuni_as2.run" in script
