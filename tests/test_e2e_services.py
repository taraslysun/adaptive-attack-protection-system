"""L2 — external services E2E. Requires `docker compose up qdrant sandbox`.

Marked `pytest.mark.services`; default-skipped.

Run:
    docker compose up -d qdrant sandbox
    pytest -m services tests/test_e2e_services.py -q
"""
from __future__ import annotations

import os
import socket
import subprocess
import urllib.error
import urllib.request

import pytest

pytestmark = pytest.mark.services

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


def _qdrant_alive() -> bool:
    try:
        with urllib.request.urlopen(f"{QDRANT_URL}/healthz", timeout=2) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _sandbox_alive() -> bool:
    try:
        out = subprocess.run(
            ["docker", "exec", "aaps-sandbox", "echo", "ok"],
            capture_output=True, text=True, timeout=3,
        )
        return out.returncode == 0 and "ok" in out.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


@pytest.fixture(scope="session", autouse=True)
def _require_services():
    missing: list[str] = []
    if not _qdrant_alive():
        missing.append(f"qdrant @ {QDRANT_URL}")
    if not _sandbox_alive():
        missing.append("sandbox container `aaps-sandbox`")
    if missing:
        pytest.skip("services unreachable: " + ", ".join(missing))


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------


def test_qdrant_health():
    """Smoke health check."""
    with urllib.request.urlopen(f"{QDRANT_URL}/healthz", timeout=2) as r:
        assert r.status == 200


def test_qdrant_collection_round_trip():
    """Create a tmp collection, write a point, query, drop."""
    import json
    import urllib.request as u

    coll = "aaps_smoke_test"
    payload = {
        "vectors": {"size": 4, "distance": "Cosine"},
    }
    req = u.Request(
        f"{QDRANT_URL}/collections/{coll}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="PUT",
    )
    with u.urlopen(req, timeout=5) as r:
        assert r.status in (200, 201)

    # Upsert one point
    point = {
        "points": [{"id": 1, "vector": [0.1, 0.2, 0.3, 0.4], "payload": {"text": "hello"}}]
    }
    req = u.Request(
        f"{QDRANT_URL}/collections/{coll}/points?wait=true",
        data=json.dumps(point).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="PUT",
    )
    with u.urlopen(req, timeout=5) as r:
        assert r.status in (200, 201)

    # Search
    search = {"vector": [0.1, 0.2, 0.3, 0.4], "limit": 1}
    req = u.Request(
        f"{QDRANT_URL}/collections/{coll}/points/search",
        data=json.dumps(search).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with u.urlopen(req, timeout=5) as r:
        body = json.loads(r.read())
        assert len(body["result"]) == 1

    # Drop
    req = u.Request(f"{QDRANT_URL}/collections/{coll}", method="DELETE")
    with u.urlopen(req, timeout=5) as r:
        assert r.status == 200


def test_memory_manager_round_trip():
    """Higher-level MemoryManager round-trip."""
    pytest.importorskip("qdrant_client")
    from aaps.agent.memory_manager import MemoryManager

    mm = MemoryManager(collection="aaps_smoke_mm", url=QDRANT_URL)
    mm.write("hello world")
    out = mm.search("hello", k=1)
    assert out and "hello" in out[0]["text"]
    mm.clear()


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------


def test_sandbox_exec_echo():
    out = subprocess.run(
        ["docker", "exec", "aaps-sandbox", "echo", "hello"],
        capture_output=True, text=True, timeout=5,
    )
    assert out.returncode == 0
    assert "hello" in out.stdout


def test_sandbox_python_available():
    out = subprocess.run(
        ["docker", "exec", "aaps-sandbox", "python3", "--version"],
        capture_output=True, text=True, timeout=5,
    )
    assert out.returncode == 0
    assert "Python" in out.stdout
