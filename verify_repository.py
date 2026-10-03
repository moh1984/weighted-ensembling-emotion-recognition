#!/usr/bin/env python3
from pathlib import Path
import hashlib, sys

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "PUBLIC_RELEASE_MANIFEST.sha256"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

if not MANIFEST.exists():
    raise SystemExit("Missing PUBLIC_RELEASE_MANIFEST.sha256")

count = 0
for raw in MANIFEST.read_text(encoding="utf-8").splitlines():
    if not raw.strip():
        continue
    digest, rel = raw.split(None, 1)
    p = ROOT / rel.strip()
    if not p.exists():
        raise SystemExit(f"MISSING: {rel.strip()}")
    got = sha256(p)
    if got != digest:
        raise SystemExit(f"HASH MISMATCH: {rel.strip()}\n expected={digest}\n actual  ={got}")
    count += 1
print(f"OK: {count} repository files verified")
