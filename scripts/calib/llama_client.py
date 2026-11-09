"""Minimal llama-server client (stdlib only) returning server-side timings."""

from __future__ import annotations

import json
import time
import urllib.request


def completion(
    prompt: str, n_predict: int, port: int = 8099, timeout: float = 7200.0, **extra
) -> dict:
    body = {
        "prompt": prompt,
        "n_predict": n_predict,
        "cache_prompt": True,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": False,
        **extra,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    out["wall_s"] = time.time() - t0
    return out


def tokenize(text: str, port: int = 8099) -> list[int]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/tokenize",
        data=json.dumps({"content": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["tokens"]


def wait_ready(port: int = 8099, timeout: float = 300.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError("llama-server not ready")
