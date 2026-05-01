#!/usr/bin/env python3
"""Static tripwire: verify no active (non-_legacy) Python file calls Ollama localhost.

Exits 0 (pass) when every ``requests.post(.../api/chat...)`` to localhost
is inside ``attacks/_legacy/`` or ``defenses/_legacy/``.
Exits 1 (fail) and prints violations otherwise.

Usage:
    python scripts/setup/verify_no_ollama.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_OLLAMA_PATTERNS = [
    re.compile(r"""requests\.post\s*\(\s*f?['"].*(?:localhost|127\.0\.0\.1).*api/chat"""),
    re.compile(r"""requests\.get\s*\(\s*f?['"].*(?:localhost|127\.0\.0\.1).*api/tags"""),
]

_LEGACY_DIRS = {"_legacy"}


def _is_legacy(path: Path) -> bool:
    return any(part in _LEGACY_DIRS for part in path.parts)


def main() -> int:
    violations: list[tuple[Path, int, str]] = []

    for py in PROJECT_ROOT.rglob("*.py"):
        if _is_legacy(py.relative_to(PROJECT_ROOT)):
            continue
        if "test" in py.name and py.name.startswith("test_"):
            continue
        try:
            lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines, 1):
            for pat in _OLLAMA_PATTERNS:
                if pat.search(line):
                    violations.append((py.relative_to(PROJECT_ROOT), i, line.strip()))

    if not violations:
        print("PASS: no active-code Ollama localhost calls found.")
        return 0

    print(f"FAIL: {len(violations)} Ollama localhost call(s) in active code:\n")
    for path, lineno, text in violations:
        print(f"  {path}:{lineno}  {text[:120]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
