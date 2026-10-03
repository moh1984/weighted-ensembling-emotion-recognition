#!/usr/bin/env python3
from pathlib import Path
import hashlib, json, sys

ROOT = Path(__file__).resolve().parent
CP = ROOT / "partA" / "predictions_checkpoint45"
CFG = ROOT / "repository" / "current_working_project" / "partA_analysis" / "partA_analysis_config.json"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

manifest = CP / "checkpoint_manifest.json"
checks = CP / "CHECKPOINT.sha256"
if not manifest.exists() or not checks.exists():
    raise SystemExit("Missing Part-A checkpoint manifest/checksum file")

cfg = json.loads(CFG.read_text(encoding="utf-8"))
bind = cfg["bindings"]
if sha256(manifest) != bind["checkpoint_manifest_sha256"]:
    raise SystemExit("Part-A checkpoint manifest does not match frozen analysis binding")
if sha256(checks) != bind["checkpoint_checksums_sha256"]:
    raise SystemExit("Part-A CHECKPOINT.sha256 does not match frozen analysis binding")

lines = [x.strip() for x in checks.read_text(encoding="utf-8").splitlines() if x.strip()]
if len(lines) != 90:
    raise SystemExit(f"Expected 90 Part-A checksum entries, found {len(lines)}")
for raw in lines:
    digest, rel = raw.split(None, 1)
    p = CP / rel.strip()
    if not p.exists():
        raise SystemExit(f"MISSING: {rel.strip()}")
    got = sha256(p)
    if got != digest:
        raise SystemExit(f"HASH MISMATCH: {rel.strip()}\n expected={digest}\n actual  ={got}")

csvs = list(CP.glob("*.csv"))
metas = list(CP.glob("*.meta.json"))
if len(csvs) != 45 or len(metas) != 45:
    raise SystemExit(f"Expected 45 CSV + 45 META; found {len(csvs)} + {len(metas)}")

for mp in metas:
    m = json.loads(mp.read_text(encoding="utf-8"))
    rid = m["run_id"]
    csvp = CP / f"{rid}.csv"
    if not csvp.exists() or sha256(csvp) != m["prediction_file_sha256"]:
        raise SystemExit(f"CSV/meta binding failure: {rid}")

cm = json.loads(manifest.read_text(encoding="utf-8"))
if cm.get("completed_runs") != 45 or cm.get("expected_runs") != 45:
    raise SystemExit("Checkpoint manifest does not report 45/45 completed runs")
print("OK: Part-A checkpoint verified (45 prediction CSV + 45 metadata files; 90/90 checksums valid)")
print("OK: Frozen Part-A analysis bindings match the embedded checkpoint")
