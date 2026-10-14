"""Local OpenAI-compatible shim in front of a native Ollama /api/chat endpoint.

Why: CAMEL (and therefore OASIS) talks to models through the OpenAI
chat-completions API (`/v1/chat/completions`). Our Ollama server is behind a
proxy that only forwards `/api/*`, so this shim translates requests and
responses. It never repairs malformed tool calls: what the model returns is
what OASIS sees, so parse failures stay countable. The only retry is for
connection failures, where the request never reached the model.

If the proxy starts forwarding `/v1/*`, point the run config straight at it
and stop using this file.

Usage:
    python -m kanuni.ollama_shim --port 8765
Env:
    KANUNI_OLLAMA_BASE   e.g. https://host.example  (no trailing /api)
    KANUNI_OLLAMA_TOKEN  bearer token for the proxy
    KANUNI_OLLAMA_THINK  "false" (default) disables thinking mode
    KANUNI_OLLAMA_NUM_CTX context window (default 32768). Ollama silently
                         truncates longer prompts, dropping the persona, so
                         the shim refuses them (context_overflow) instead.
    KANUNI_SHIM_LOG      optional path; raw request/response pairs as JSONL
"""
import argparse
import asyncio
import json
import os
import time
import uuid

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from kanuni.env_file import load_env_file

load_env_file()
BASE = os.environ.get("KANUNI_OLLAMA_BASE", "").rstrip("/")
TOKEN = os.environ.get("KANUNI_OLLAMA_TOKEN", "")
THINK = os.environ.get("KANUNI_OLLAMA_THINK", "false").lower() == "true"
NUM_CTX = int(os.environ.get("KANUNI_OLLAMA_NUM_CTX", "32768"))
# Room kept for the reply. Prompts estimated above NUM_CTX - REPLY_RESERVE
# are refused rather than sent, because Ollama would cut their start.
REPLY_RESERVE = 2048
# Measured on qwen3.5 prompts (messages + tool schemas): 3.0-3.4 chars per
# token. 3.0 over-estimates, so borderline prompts are refused, not cut.
CHARS_PER_TOKEN = 3.0
LOG_PATH = os.environ.get("KANUNI_SHIM_LOG")
CONNECT_RETRIES = 3

client = httpx.AsyncClient(timeout=httpx.Timeout(600.0))


def _text(content) -> str:
    # OpenAI content may be a list of parts; Ollama wants a string.
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content
                   if isinstance(p, dict) and p.get("type") == "text")


def to_ollama_messages(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        msg = {"role": m["role"], "content": _text(m.get("content"))}
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                args = c["function"].get("arguments") or "{}"
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                calls.append({"function": {"name": c["function"]["name"],
                                           "arguments": args}})
            msg["tool_calls"] = calls
        if m["role"] == "tool" and m.get("name"):
            msg["tool_name"] = m["name"]
        out.append(msg)
    return out


def to_openai_response(body: dict, model: str) -> dict:
    msg = body.get("message", {})
    tool_calls = []
    for c in msg.get("tool_calls") or []:
        fn = c.get("function", {})
        args = fn.get("arguments", {})
        tool_calls.append({
            "id": c.get("id") or f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": fn.get("name", ""),
                "arguments": args if isinstance(args, str) else json.dumps(
                    args, ensure_ascii=False),
            },
        })
    message = {"role": "assistant", "content": msg.get("content") or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    prompt_tokens = body.get("prompt_eval_count", 0)
    completion_tokens = body.get("eval_count", 0)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if tool_calls else (
                "length" if body.get("done_reason") == "length" else "stop"),
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def estimate_tokens(payload: dict) -> int:
    chars = sum(len(m.get("content") or "") + len(json.dumps(
        m.get("tool_calls") or [])) for m in payload["messages"])
    chars += len(json.dumps(payload.get("tools") or []))
    return int(chars / CHARS_PER_TOKEN)


def _upstream_error(message: str) -> JSONResponse:
    print(f"upstream error: {message}", flush=True)
    return JSONResponse({"error": {"message": message}}, status_code=502)


async def chat_completions(request: Request):
    req = await request.json()
    if req.get("stream"):
        return JSONResponse({"error": {"message": "stream not supported"}},
                            status_code=400)
    options = {"num_ctx": NUM_CTX}
    for src, dst in [("temperature", "temperature"), ("top_p", "top_p"),
                     ("seed", "seed"), ("max_tokens", "num_predict")]:
        if req.get(src) is not None:
            options[dst] = req[src]
    payload = {
        "model": req["model"],
        "messages": to_ollama_messages(req["messages"]),
        "stream": False,
        "think": THINK,
        "options": options,
    }
    if req.get("tools"):
        payload["tools"] = req["tools"]
    est = estimate_tokens(payload)
    if est > NUM_CTX - REPLY_RESERVE:
        msg = (f"context_overflow: prompt ~{est} tokens exceeds the "
               f"{NUM_CTX}-token window minus {REPLY_RESERVE} for the reply; "
               "not sent, because Ollama would silently cut its start")
        print(msg, flush=True)
        return JSONResponse({"error": {"message": msg,
                                       "type": "context_overflow"}},
                            status_code=400)
    # Retry only when the request never reached the server (DNS/connect
    # failure): no model output exists yet, so nothing is hidden. Anything
    # else (timeouts, upstream errors) is returned as a 502 and counted.
    for attempt in range(1, CONNECT_RETRIES + 2):
        try:
            resp = await client.post(
                f"{BASE}/api/chat", json=payload,
                headers={"Authorization": f"Bearer {TOKEN}"})
            break
        except httpx.ConnectError as e:
            if attempt > CONNECT_RETRIES:
                return _upstream_error(f"connect failed: {e}")
            print(f"connect retry {attempt}: {e}", flush=True)
            await asyncio.sleep(2 ** attempt)
        except httpx.HTTPError as e:
            return _upstream_error(f"{type(e).__name__}: {e}")
    if resp.status_code != 200:
        return JSONResponse(
            {"error": {"message": f"upstream {resp.status_code}: "
                       f"{resp.text[:500]}"}},
            status_code=502)
    body = resp.json()
    # prompt_eval_count excludes cached tokens and is measured after any
    # truncation, so it cannot show a cut on its own; compare with the
    # estimate as a second check.
    seen = body.get("prompt_eval_count", 0) + body.get(
        "prompt_eval_cached_count", 0)
    if seen >= NUM_CTX - 16 or (est > NUM_CTX // 2 and seen < 0.6 * est):
        print(f"WARNING: prompt may have been truncated (estimated ~{est} "
              f"tokens, model saw {seen}).", flush=True)
    out = to_openai_response(body, req["model"])
    if LOG_PATH:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"request": payload, "response": body},
                               ensure_ascii=False) + "\n")
    return JSONResponse(out)


async def models(request: Request):
    resp = await client.get(f"{BASE}/api/tags",
                            headers={"Authorization": f"Bearer {TOKEN}"})
    names = [m["name"] for m in resp.json().get("models", [])]
    return JSONResponse({"object": "list", "data": [
        {"id": n, "object": "model", "owned_by": "ollama"} for n in names]})


app = Starlette(routes=[
    Route("/v1/chat/completions", chat_completions, methods=["POST"]),
    Route("/v1/models", models, methods=["GET"]),
])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not BASE or not TOKEN:
        raise SystemExit("Set KANUNI_OLLAMA_BASE and KANUNI_OLLAMA_TOKEN "
                         "(e.g. in kanuni/.env).")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
