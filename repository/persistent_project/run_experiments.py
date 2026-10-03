#!/usr/bin/env python3
"""
Protocol v7 - run_experiments.py (Part A, Phase 1)

Trains the 45 Part-A runs and writes one prediction file per run. Every run
records the SHA-256 of the split index files it actually read, so the analysis
can prove it used the frozen splits rather than a rebuild.

Designed to be interrupted. A finished run is skipped on the next invocation,
so a session limit costs at most the one run that was in flight.

    # one split per session keeps each session comfortably inside a time limit
    python run_experiments.py --splits-dir ./splits --out-dir ./predictions \\
                              --split-seed 42

    # resume, or cap a session
    python run_experiments.py --splits-dir ./splits --out-dir ./predictions \\
                              --split-seed 42 --max-runs 6

Prediction file columns, fixed by the protocol:
    example_id, true_label, pred_label, p_anger, p_disgust, p_fear,
    p_joy, p_sadness, p_surprise
"""

import argparse
import hashlib
import re
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

EKMAN = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
LABEL2ID = {l: i for i, l in enumerate(EKMAN)}

# ---------------------------------------------------------------------------
# Numeric training settings are read from the frozen training_config.json and
# cannot be overridden from the command line: a value that can be changed per
# invocation is not frozen, whatever a document says.
#
# Some settings are structural rather than numeric - the optimizer, scheduler,
# loss, padding strategy and checkpoint rule are implemented directly in this
# file. Writing "optimizer": "SGD" in the config would NOT make this runner use
# SGD. Rather than making the runner generic, it refuses any config whose
# structural fields disagree with what it actually executes, so the config
# never describes behaviour the code does not have.
# ---------------------------------------------------------------------------

# (section, key) -> the value this runner actually implements.
IMPLEMENTED = {
    ("tokenization", "padding"): "max_length",
    ("tokenization", "truncation"): True,
    ("optimization", "optimizer"): "AdamW",
    ("optimization", "scheduler"): "linear_with_warmup",
    ("optimization", "loss"): "cross_entropy",
    ("checkpoint_rule", "selection_metric"): "validation_macro_f1",
    ("checkpoint_rule", "selection_mode"): "max",
    ("checkpoint_rule", "evaluate_every"): "epoch",
    ("checkpoint_rule", "restore_best_before_test"): True,
}

FROZEN_FIELDS = ("class_weighting", "models", "tokenization", "optimization",
                 "checkpoint_rule", "model_seeds", "labels", "determinism",
                 "data_provenance", "h2_single_model_baseline")


def load_config(path: Path):
    cfg = json.loads(Path(path).read_text())
    missing = [f for f in FROZEN_FIELDS if f not in cfg]
    if missing:
        raise SystemExit(f"[FATAL] training config is missing {missing}")
    if not cfg.get("frozen"):
        raise SystemExit(
            "[FATAL] training config is not marked frozen. Set \"frozen\": true "
            "only when every value is final; Phase 1 results are invalid "
            "otherwise.")
    for name, spec in cfg["models"].items():
        rev = str(spec.get("revision", ""))
        if not re.fullmatch(r"[0-9a-fA-F]{40}", rev):
            raise SystemExit(
                f"[FATAL] model '{name}' revision {rev!r} is not a full "
                "40-character commit SHA. Branch names and tags are mutable "
                "and pin nothing.")
    if list(cfg["labels"]) != EKMAN:
        raise SystemExit("[FATAL] config labels do not match the Ekman order.")

    for (section, key), want in IMPLEMENTED.items():
        got = cfg.get(section, {}).get(key)
        if got != want:
            raise SystemExit(
                f"[FATAL] {section}.{key}={got!r} is not what this runner "
                f"implements ({want!r}). Structural settings are hard-coded; "
                "the config may describe them but may not contradict them.")

    if cfg["class_weighting"] != "none":
        raise SystemExit(
            "[FATAL] the Phase-1 runner implements unweighted cross-entropy "
            f"only; class_weighting={cfg['class_weighting']!r} is not "
            "executable here.")

    if cfg["determinism"].get("seed_python_numpy_torch") is not True:
        raise SystemExit(
            "[FATAL] the Phase-1 runner always seeds Python, NumPy and Torch; "
            "seed_python_numpy_torch must be true so the config does not "
            "describe a behaviour the code does not offer.")

    dp = cfg["data_provenance"]
    for f in ("dataset_manifest_sha256", "corpus_content_sha256", "split_seeds"):
        v = dp.get(f)
        if not v or (isinstance(v, str) and v.startswith("<")):
            raise SystemExit(
                f"[FATAL] data_provenance.{f} is unset or still a placeholder. "
                "A frozen config must name the exact dataset it belongs to.")

    b = cfg["h2_single_model_baseline"]
    if (b.get("model") not in cfg["models"] or b.get("regime") != "r_clean"
            or b.get("model_seed") not in cfg["model_seeds"]):
        raise SystemExit(
            "[FATAL] h2_single_model_baseline must name a trained "
            "model/regime/seed combination.")

    return cfg, hashlib.sha256(Path(path).read_bytes()).hexdigest()




def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def apply_determinism(det):
    """Apply the declared determinism settings rather than merely recording
    them. A config that states a setting the code never reads documents an
    intention, not a behaviour."""
    torch.backends.cudnn.benchmark = bool(det.get("cudnn_benchmark", True))
    want = bool(det.get("torch_use_deterministic_algorithms", False))
    if want:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(want, warn_only=True)
    return {"cudnn_benchmark": torch.backends.cudnn.benchmark,
            "torch_use_deterministic_algorithms": want}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_corpus(splits_dir: Path):
    for name in ("corpus.parquet", "corpus.csv.gz"):
        p = splits_dir / name
        if p.exists():
            df = (pd.read_parquet(p) if p.suffix == ".parquet"
                  else pd.read_csv(p))
            return df.set_index("id", drop=False), name
    raise FileNotFoundError(f"no corpus file in {splits_dir}")


def read_ids(path: Path):
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]


class TextDS(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.enc = tok(list(texts), truncation=True, max_length=max_len,
                       padding="max_length", return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return ({k: v[i] for k, v in self.enc.items()}, self.labels[i])


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------

def macro_f1(y_true, y_pred, n=len(EKMAN)):
    f1s = []
    for c in range(n):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s)), f1s


def train_one(train_df, val_df, test_df, backbone, seed, class_weights,
              device, cfg):
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              get_linear_schedule_with_warmup)
    set_seed(seed)
    spec = cfg["models"][backbone]
    tk, op, ck = cfg["tokenization"], cfg["optimization"], cfg["checkpoint_rule"]
    tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    model = AutoModelForSequenceClassification.from_pretrained(
        spec["repo"], revision=spec["revision"],
        num_labels=len(EKMAN)).to(device)
    # Fail closed: an unverifiable revision is not a pinned one. Skipping the
    # check when _commit_hash is absent would silently accept whatever the hub
    # served.
    resolved = getattr(model.config, "_commit_hash", None)
    if not resolved:
        raise SystemExit(
            f"[FATAL] {backbone}: the loaded checkpoint's commit could not be "
            "read back, so the pinned revision cannot be verified. Upgrade "
            "transformers/huggingface_hub before a frozen run.")
    if str(resolved).lower() != spec["revision"].lower():
        raise SystemExit(
            f"[FATAL] {backbone}: requested revision {spec['revision']}, "
            f"loaded {resolved}.")

    def mk(df, bs, shuffle):
        ds = TextDS(df["text"].astype(str).tolist(),
                    [LABEL2ID[l] for l in df["label"]], tok, tk["max_length"])
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=2)

    tr = mk(train_df, op["batch_size"], True)
    va = mk(val_df, op["eval_batch_size"], False)
    te = mk(test_df, op["eval_batch_size"], False)

    opt = torch.optim.AdamW(model.parameters(), lr=op["learning_rate"],
                            weight_decay=op["weight_decay"],
                            eps=op.get("adam_epsilon", 1e-8))
    total = len(tr) * op["epochs"]
    sch = get_linear_schedule_with_warmup(
        opt, int(op["warmup_ratio"] * total), total)
    w = (torch.tensor(class_weights, dtype=torch.float, device=device)
         if class_weights is not None else None)

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        P, Y = [], []
        for batch, y in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            P.append(F.softmax(model(**batch).logits, dim=-1).cpu().numpy())
            Y.append(y.numpy())
        return np.concatenate(P), np.concatenate(Y)

    best_f1, best_state, patience = -1.0, None, 0
    for ep in range(op["epochs"]):
        model.train()
        for batch, y in tr:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = F.cross_entropy(model(**batch).logits, y.to(device), weight=w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           op["max_grad_norm"])
            opt.step(); sch.step(); opt.zero_grad()

        vp, vy = evaluate(va)
        f1, _ = macro_f1(vy, vp.argmax(1))
        print(f"      epoch {ep+1}: val macro-F1 {f1:.4f}")
        if f1 > best_f1:
            best_f1, patience = f1, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= ck["early_stopping_patience"]:
                print("      early stop")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    tp, ty = evaluate(te)
    va_p, va_y = evaluate(va)
    val_f1, _ = macro_f1(va_y, va_p.argmax(1))
    return tp, ty, val_f1, str(resolved)


# ---------------------------------------------------------------------------
# Job definition: the 15 Part-A runs of one split instance
# ---------------------------------------------------------------------------

def build_jobs(split_dir: Path, split_seed: int, model_seeds):
    jury = json.loads((split_dir / "jury_blocks.json").read_text())
    blocks = jury["blocks"]
    jobs = []

    for backbone in ("bert", "roberta"):
        for ms in model_seeds:
            jobs.append({"model": backbone, "regime": "r_clean",
                         "train_file": "r_clean_train.txt",
                         "model_seed": ms, "split_seed": split_seed,
                         "author_blocks": None})
    for ms in model_seeds:                       # H1 vehicle
        jobs.append({"model": "roberta", "regime": "r_overlap",
                     "train_file": "r_overlap_train.txt",
                     "model_seed": ms, "split_seed": split_seed,
                     "author_blocks": None})

    for j, pair in (("J1", ["A", "B"]), ("J2", ["A", "C"]), ("J3", ["B", "C"])):
        jobs.append({"model": "bert", "regime": f"complementary_{j}",
                     "train_file": "r_clean_train.txt",
                     "model_seed": model_seeds[0], "split_seed": split_seed,
                     "author_blocks": [blocks[b] for b in pair]})
    for b in ("A", "B", "C"):
        jobs.append({"model": "bert", "regime": f"disjoint_{b}",
                     "train_file": "r_clean_train.txt",
                     "model_seed": model_seeds[0], "split_seed": split_seed,
                     "author_blocks": [blocks[b]]})
    return jobs


def is_complete(out_csv: Path, meta_path: Path, expected_run_id: str,
                cfg_sha: str, n_test: int, runner_sha: str, manifest_sha: str):
    """A run counts as finished only if BOTH files exist, the metadata cites
    the current config, and the prediction file has the expected number of
    rows. A session killed mid-write leaves a truncated CSV; treating its mere
    existence as completion would silently admit a partial run into the
    analysis."""
    if not (out_csv.exists() and meta_path.exists()):
        return False, "missing file"
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return False, "unreadable metadata"
    # Every other check describes the config, the code or the data - all
    # shared across runs. Without this one, a CSV/metadata pair copied from a
    # different run and renamed would satisfy them all. That is a live risk in
    # the download -> dataset -> copy cycle, so identity is checked first.
    if meta.get("run_id") != expected_run_id:
        return False, "run id mismatch"
    if meta.get("training_config_sha256") != cfg_sha:
        return False, "built with a different training config"
    if meta.get("run_experiments_sha256") != runner_sha:
        return False, "produced by a different version of the runner"
    if meta.get("dataset_manifest_sha256") != manifest_sha:
        return False, "produced from a different dataset"
    # Row count proves completeness, not identity: a file from another run with
    # the same length would pass. In the download -> dataset -> copy -> resume
    # cycle that is a real possibility, so the bytes are checked.
    expected_pred = meta.get("prediction_file_sha256")
    if not expected_pred:
        return False, "missing prediction checksum"
    if sha_file(out_csv) != expected_pred:
        return False, "prediction checksum mismatch"
    if meta.get("n_test") != n_test:
        return False, "test size disagrees with the split"
    try:
        rows = sum(1 for _ in out_csv.open()) - 1
    except Exception:
        return False, "unreadable predictions"
    if rows != n_test:
        return False, f"truncated predictions ({rows}/{n_test})"
    return True, "ok"


def atomic_write(path: Path, write_fn):
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_fn(tmp)
    os.replace(tmp, path)


def run_id(job):
    return (f"{job['model']}__{job['regime']}__split{job['split_seed']}"
            f"__seed{job['model_seed']}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--split-seed", type=int, action="append",
                    help="Repeatable. Default: all three.")
    ap.add_argument("--preflight", action="store_true",
                    help="Validate config, splits, tokenizers and provenance "
                         "without training and without touching the test set.")
    ap.add_argument("--max-runs", type=int, default=None,
                    help="Stop after this many NEW runs this session. Skips "
                         "are not counted, so --max-runs 0 reports what is "
                         "already complete and trains nothing.")
    ap.add_argument("--config", type=Path, required=True,
                    help="Frozen training_config.json. All training settings "
                         "come from here; there are no CLI overrides.")
    args = ap.parse_args()

    cfg, cfg_sha = load_config(args.config)
    runner_sha = sha_file(__file__)
    det_applied = apply_determinism(cfg["determinism"])
    print(f"[cfg  ] {args.config.name} sha256={cfg_sha[:16]} "
          f"version={cfg.get('config_version')} "
          f"class_weighting={cfg['class_weighting']}")
    print(f"[code ] run_experiments.py sha256={runner_sha[:16]}")
    print(f"[det  ] {det_applied}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[warn ] no GPU detected; this will be very slow.")

    manifest_path = args.splits_dir / "dataset_manifest.json"
    manifest_sha = sha_file(manifest_path)
    manifest = json.loads(manifest_path.read_text())

    # Bind config to dataset BEFORE training, not merely record it afterwards.
    dp = cfg["data_provenance"]
    if not manifest.get("freeze_quality"):
        raise SystemExit(
            "[FATAL] these splits are not a frozen dataset "
            "(freeze_quality is false). Phase-1 runs require frozen splits.")
    if manifest_sha.lower() != str(dp["dataset_manifest_sha256"]).lower():
        raise SystemExit(
            "[FATAL] dataset manifest mismatch.\n"
            f"  config expects : {dp['dataset_manifest_sha256']}\n"
            f"  splits on disk : {manifest_sha}\n"
            "This config is frozen against one specific dataset.")
    if manifest.get("corpus_content_sha256", "").lower() != \
            str(dp["corpus_content_sha256"]).lower():
        raise SystemExit("[FATAL] corpus content hash does not match the config.")
    print(f"[data ] manifest sha256={manifest_sha[:16]} verified against config")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    allowed = set(cfg["data_provenance"]["split_seeds"])
    seeds = args.split_seed or sorted(allowed)
    bad = [s2 for s2 in seeds if s2 not in allowed]
    if bad:
        raise SystemExit(
            f"[FATAL] split seed(s) {bad} are not in the frozen set "
            f"{sorted(allowed)}.")

    if args.preflight:
        from transformers import AutoTokenizer, AutoConfig
        for name, spec in cfg["models"].items():
            AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
            # Resolve and verify the commit without downloading the weights,
            # so preflight checks the same thing training will.
            mc = AutoConfig.from_pretrained(spec["repo"],
                                            revision=spec["revision"])
            resolved = getattr(mc, "_commit_hash", None)
            if not resolved:
                raise SystemExit(
                    f"[FATAL] {name}: revision could not be resolved in "
                    "preflight; a frozen run would fail the same check.")
            if str(resolved).lower() != spec["revision"].lower():
                raise SystemExit(
                    f"[FATAL] {name}: requested {spec['revision']}, "
                    f"resolved {resolved}")
            print(f"[pre  ] {name}: tokenizer ok, revision "
                  f"{str(resolved)[:12]} requested == resolved")
        for sseed in seeds:
            sd = args.splits_dir / f"seed{sseed}"
            jobs = build_jobs(sd, sseed, cfg["model_seeds"])
            n_tr = len(read_ids(sd / "r_clean_train.txt"))
            print(f"[pre  ] seed {sseed}: {len(jobs)} jobs, "
                  f"train={n_tr:,}, blocks ok")
        print("[pre  ] preflight passed: no corpus loaded, no model trained, "
              "test set untouched")
        return

    # Loaded only for real runs. Keeping it out of the preflight path is what
    # makes "the test set is untouched" literally true rather than merely true
    # of inference: preflight needs the manifest, the tokenizers and the job
    # grid, and nothing else.
    corpus, corpus_file = load_corpus(args.splits_dir)
    done = 0

    for sseed in seeds:
        sd = args.splits_dir / f"seed{sseed}"
        val_ids, test_ids = read_ids(sd / "r_clean_val.txt"), read_ids(sd / "r_clean_test.txt")
        val_df, test_df = corpus.loc[val_ids], corpus.loc[test_ids]

        for job in build_jobs(sd, sseed, cfg["model_seeds"]):
            rid = run_id(job)
            out = args.out_dir / f"{rid}.csv"
            meta_path = args.out_dir / f"{rid}.meta.json"
            ok, why = is_complete(out, meta_path, rid, cfg_sha, len(test_df),
                                  runner_sha, manifest_sha)
            if ok:
                print(f"[skip ] {rid}")
                continue
            # The cap counts NEW runs, so it is evaluated after the skip check:
            # a completed run must still be reported as skipped, and
            # --max-runs 0 must report every skip and then start nothing. If
            # the cap were tested first it would exit before printing anything,
            # which is exactly what a resume check needs to see.
            if args.max_runs is not None and done >= args.max_runs:
                print(f"[stop ] session cap reached ({args.max_runs})")
                return
            if out.exists() or meta_path.exists():
                print(f"[redo ] {rid}  ({why}) - discarding partial output")
                for f in (out, meta_path, out.with_suffix(".csv.tmp"),
                          meta_path.with_suffix(".json.tmp")):
                    if f.exists():
                        f.unlink()
            train_path = sd / job["train_file"]
            train_ids = read_ids(train_path)
            train_df = corpus.loc[train_ids]
            if job["author_blocks"] is not None:
                keep = set().union(*[set(b) for b in job["author_blocks"]])
                train_df = train_df[train_df["author"].isin(keep)]

            cw = None
            if cfg["class_weighting"] == "balanced":
                counts = train_df["label"].value_counts().reindex(EKMAN).fillna(0)
                cw = (len(train_df) / (len(EKMAN) * counts.clip(lower=1))).tolist()

            print(f"[run  ] {rid}  train={len(train_df):,} "
                  f"val={len(val_df):,} test={len(test_df):,}")
            t0 = time.time()
            probs, ytrue, val_f1, resolved_rev = train_one(
                train_df, val_df, test_df, job["model"], job["model_seed"],
                cw, device, cfg)
            elapsed = round(time.time() - t0, 1)

            preds = pd.DataFrame({
                "example_id": test_df["id"].tolist(),
                "true_label": [EKMAN[i] for i in ytrue],
                "pred_label": [EKMAN[i] for i in probs.argmax(1)],
                **{f"p_{c}": probs[:, i] for i, c in enumerate(EKMAN)},
            })
            atomic_write(out, lambda t: preds.to_csv(t, index=False))
            prediction_sha = sha_file(out)

            # Provenance: the index-file hashes prove which splits were read.
            meta_payload = json.dumps({
                "run_id": rid,
                "model": job["model"],
                "regime": job["regime"],
                "train_file": job["train_file"],
                "model_seed": job["model_seed"],
                "split_seed": job["split_seed"],
                # author_blocks is deliberately NOT copied here: for jury runs
                # it lists thousands of usernames, and the jury_blocks.json
                # hash plus effective_train_ids_sha256 already identify the
                # training set exactly.
                "uses_author_blocks": job["author_blocks"] is not None,
                # Full precision: the H3 jury weights are softmax over these
                # validation scores, so a rounded value would not reproduce the
                # weights actually used - and rounding bites hardest exactly
                # when jurors are close, which is the case H3 examines.
                "validation_macro_f1": float(val_f1),
                "validation_macro_f1_display": round(float(val_f1), 6),
                "n_train": int(len(train_df)), "n_val": int(len(val_df)),
                "n_test": int(len(test_df)),
                "prediction_file_sha256": prediction_sha,
                "class_weighting": cfg["class_weighting"],
                "training_config_sha256": cfg_sha,
                "training_config_version": cfg.get("config_version"),
                "run_experiments_sha256": runner_sha,
                "dataset_manifest_sha256": manifest_sha,
                "determinism_applied": det_applied,
                "backbone_repo": cfg["models"][job["model"]]["repo"],
                "backbone_revision_requested":
                    cfg["models"][job["model"]]["revision"],
                "backbone_revision_resolved": resolved_rev,
                "index_file_sha256": {
                    job["train_file"]: sha_file(train_path),
                    "r_clean_val.txt": sha_file(sd / "r_clean_val.txt"),
                    "r_clean_test.txt": sha_file(sd / "r_clean_test.txt"),
                    "jury_blocks.json": sha_file(sd / "jury_blocks.json"),
                },
                # The exact examples this run saw. For jury members the index
                # file alone is not enough: the author-block filter decides the
                # training set, so the post-filter ids are hashed directly.
                "effective_train_ids_sha256": hashlib.sha256(
                    ("\n".join(train_df["id"].astype(str)) + "\n")
                    .encode("utf-8")).hexdigest(),
                "corpus_file": corpus_file,
                "corpus_content_sha256": manifest.get("corpus_content_sha256"),
                "splits_freeze_quality": manifest.get("freeze_quality"),
                "elapsed_seconds": elapsed,
                "environment": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "transformers": __import__("transformers").__version__,
                    "device": torch.cuda.get_device_name(0)
                              if device == "cuda" else "cpu",
                },
            }, indent=2)
            atomic_write(meta_path, lambda t: t.write_text(meta_payload))
            done += 1
            print(f"[ok   ] {rid}  val macro-F1 {val_f1:.4f}  ({elapsed}s)")

    print(f"\n[done ] {done} run(s) this session; "
          f"{len(list(args.out_dir.glob('*.csv')))} prediction files total")


if __name__ == "__main__":
    main()
