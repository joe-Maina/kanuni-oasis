"""The simulation gateway: the Kanuni backend submits runs to this
server, follows them and collects their results over HTTPS, instead of
over SSH.

    POST /v1/jobs                    submit a run (idempotent per run id)
    GET  /v1/jobs/{run_id}           its state
    POST /v1/jobs/{run_id}/cancel    stop it
    GET  /v1/jobs/{run_id}/log       the last lines of its log
    GET  /v1/jobs/{run_id}/results   its result files, as a .tar.gz
    GET  /healthz                    unauthenticated liveness

Every /v1 request carries the API key (auth.py). Run it behind a TLS proxy:
    uvicorn kanuni_gateway.app:app --host 127.0.0.1 --port 8700
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from kanuni_gateway import auth, jobs
from kanuni_gateway.settings import GatewaySettings


class SubmitRequest(BaseModel):
    run_id: str
    simulator: str
    config: dict
    # Input files, keyed by their path in the job folder (inputs/<name>.json).
    files: dict[str, str] = {}


def create_app(settings: GatewaySettings | None = None, runner: jobs.Runner | None = None) -> FastAPI:
    settings = settings or GatewaySettings()
    api_key = settings.api_key.get_secret_value()
    if len(api_key) < 32:
        raise ValueError("KANUNI_GATEWAY_API_KEY must be at least 32 characters")
    settings.root = settings.root.rstrip("/") or "/"
    runner = runner or jobs.make_runner(settings)
    submit_lock = threading.Lock()
    app = FastAPI(title="Kanuni simulation gateway", docs_url=None, redoc_url=None, openapi_url=None)

    def authorized(request: Request) -> None:
        if not auth.check(api_key, request.headers.get("authorization")):
            raise HTTPException(401, "missing or wrong API key")
        if int(request.headers.get("content-length") or 0) > settings.max_request_mb * 1024 * 1024:
            raise HTTPException(413, "request too large")

    def job(run_id: str) -> tuple[jobs.RunPaths, dict] | None:
        """The job's paths and its job.json, or None if it was never submitted."""
        try:
            jobs.check_run_id(run_id)
        except jobs.SubmissionError as exc:
            raise HTTPException(422, str(exc)) from exc
        meta_path = jobs.run_paths(settings.root, jobs.SIMULATORS[0], run_id).meta
        if not meta_path.is_file():
            return None
        meta = json.loads(meta_path.read_text())
        return jobs.run_paths(settings.root, meta["simulator"], run_id), meta

    def existing(run_id: str) -> tuple[jobs.RunPaths, dict]:
        found = job(run_id)
        if found is None:
            raise HTTPException(404, "no such job")
        return found

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @app.post("/v1/jobs", dependencies=[Depends(authorized)])
    def submit(body: SubmitRequest) -> dict:
        try:
            config = jobs.check_submission(settings, body.run_id, body.simulator, body.config, body.files)
        except jobs.SubmissionError as exc:
            raise HTTPException(422, str(exc)) from exc
        with submit_lock:
            found = job(body.run_id)
            if found is not None:
                # Already submitted (a retried request): the same job, not a second one.
                return {"job_id": body.run_id, "state": "PENDING", "created": False}
            paths = jobs.run_paths(settings.root, body.simulator, body.run_id)
            paths.job_dir.mkdir(parents=True, exist_ok=True)
            for name, text in body.files.items():
                (paths.job_dir / name).parent.mkdir(parents=True, exist_ok=True)
                (paths.job_dir / name).write_text(text)
            paths.config.write_text(json.dumps(config, indent=2, ensure_ascii=False))
            paths.script.write_text(jobs.build_job_script(settings, body.simulator, body.run_id, paths))
            try:
                native_id = runner.submit(paths)
            except jobs.RunnerError as exc:
                raise HTTPException(502, f"submission failed: {exc}") from exc
            paths.meta.write_text(json.dumps({"run_id": body.run_id, "simulator": body.simulator,
                                              "native_id": native_id, "executor": settings.executor,
                                              "submitted_at": int(time.time())}))
        return {"job_id": body.run_id, "state": "PENDING", "created": True}

    @app.get("/v1/jobs/{run_id}", dependencies=[Depends(authorized)])
    def state(run_id: str) -> dict:
        found = job(run_id)
        if found is None:
            return {"job_id": run_id, "state": jobs.MISSING}
        try:
            return {"job_id": run_id, "state": runner.state(*_native(found))}
        except jobs.RunnerError as exc:
            raise HTTPException(502, str(exc)) from exc

    @app.post("/v1/jobs/{run_id}/cancel", dependencies=[Depends(authorized)])
    def cancel(run_id: str) -> dict:
        try:
            runner.cancel(*_native(existing(run_id)))
        except jobs.RunnerError as exc:
            raise HTTPException(502, str(exc)) from exc
        return {"job_id": run_id, "cancelled": True}

    @app.get("/v1/jobs/{run_id}/log", dependencies=[Depends(authorized)], response_class=PlainTextResponse)
    def log(run_id: str, lines: int = Query(60, ge=1, le=2000)) -> str:
        paths, _ = existing(run_id)
        return jobs.tail(paths.log, lines)

    @app.get("/v1/jobs/{run_id}/results", dependencies=[Depends(authorized)])
    def results(run_id: str) -> FileResponse:
        paths, meta = existing(run_id)
        fd, name = tempfile.mkstemp(suffix=".tar.gz")
        try:
            with os.fdopen(fd, "wb") as out:
                jobs.results_archive(paths, meta["simulator"], out)
        except FileNotFoundError as exc:
            os.unlink(name)
            raise HTTPException(409, str(exc)) from exc
        digest = hashlib.sha256()
        with open(name, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        return FileResponse(name, media_type="application/gzip", filename=f"{run_id}.tar.gz",
                            headers={"X-Content-SHA256": digest.hexdigest()},
                            background=BackgroundTask(os.unlink, name))

    return app


def _native(found: tuple[jobs.RunPaths, dict]) -> tuple[jobs.RunPaths, str]:
    paths, meta = found
    return paths, str(meta["native_id"])


def __getattr__(name: str):
    # `uvicorn kanuni_gateway.app:app` builds the app from the environment
    # on first use, so importing this module in tests needs no settings.
    if name == "app":
        return create_app()
    raise AttributeError(name)
