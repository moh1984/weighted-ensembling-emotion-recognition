#!/usr/bin/env python3

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


EKMAN = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
]

LABEL2ID = {
    label: i
    for i, label in enumerate(EKMAN)
}


def sha_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def canonical_id_sha(df):

    ids = sorted(
        df["example_id"]
        .astype(str)
        .tolist()
    )

    payload = (
        "\n".join(ids) + "\n"
    ).encode("utf-8")

    return hashlib.sha256(
        payload
    ).hexdigest()


def atomic_text(path, text):

    path = Path(path)

    tmp = path.with_name(
        path.name + ".tmp"
    )

    tmp.write_text(
        text,
        encoding="utf-8",
    )

    os.replace(tmp, path)


def atomic_csv(path, df):

    path = Path(path)

    tmp = path.with_name(
        path.name + ".tmp"
    )

    df.to_csv(
        tmp,
        index=False,
    )

    os.replace(tmp, path)


def import_source(path):

    spec = (
        importlib.util
        .spec_from_file_location(
            "immutable_partA_runner",
            path,
        )
    )

    module = (
        importlib.util
        .module_from_spec(spec)
    )

    spec.loader.exec_module(
        module
    )

    return module


def verify_freeze(data_dir):

    freeze = (
        data_dir / "FREEZE.sha256"
    )

    n = 0

    for line in freeze.read_text(
        encoding="utf-8"
    ).splitlines():

        if not line.strip():
            continue

        digest, rel = line.split(
            None,
            1,
        )

        p = (
            data_dir / rel.strip()
        )

        if not p.exists():
            raise SystemExit(
                "[FATAL] frozen V2 file "
                f"missing: {rel}"
            )

        if sha_file(p) != digest:
            raise SystemExit(
                "[FATAL] frozen V2 hash "
                f"mismatch: {rel}"
            )

        n += 1

    return n


def prediction_df(frame, probs):

    idx = probs.argmax(axis=1)

    result = pd.DataFrame({
        "example_id":
            frame["example_id"]
            .astype(str)
            .tolist(),

        "true_label":
            frame["label"]
            .astype(str)
            .tolist(),

        "pred_label":
            [
                EKMAN[i]
                for i in idx
            ],
    })

    for i, label in enumerate(
        EKMAN
    ):
        result[
            f"p_{label}"
        ] = probs[:, i]

    return result


def score(source, frame, probs):

    y = np.asarray([
        LABEL2ID[x]
        for x in frame[
            "label"
        ].astype(str)
    ])

    pred = probs.argmax(axis=1)

    macro, per_class = (
        source.macro_f1(
            y,
            pred,
        )
    )

    return {
        "macro_f1":
            float(macro),

        "per_class_f1": {
            EKMAN[i]:
                float(per_class[i])
            for i in range(6)
        },
    }


def output_paths(
    out_dir,
    run_id,
):

    names = [
        "native_validation",
        "reddit_validation",
        "reddit_test",
        "twitter_test",
    ]

    return {
        name:
            out_dir
            / f"{run_id}__{name}.csv"

        for name in names
    }


def complete(
    out_dir,
    run_id,
    expected,
):

    meta_path = (
        out_dir
        / f"{run_id}.meta.json"
    )

    if not meta_path.exists():
        return False, "missing metadata"

    try:
        meta = json.loads(
            meta_path.read_text(
                encoding="utf-8"
            )
        )

    except Exception:
        return False, (
            "unreadable metadata"
        )

    if (
        meta.get("run_id")
        != run_id
    ):
        return False, (
            "run-id mismatch"
        )

    for key, value in (
        expected.items()
    ):

        if meta.get(key) != value:
            return False, (
                f"{key} mismatch"
            )

    predictions = meta.get(
        "prediction_files",
        {},
    )

    required = {
        "native_validation",
        "reddit_validation",
        "reddit_test",
        "twitter_test",
    }

    if set(predictions) != required:
        return False, (
            "prediction set mismatch"
        )

    for name in required:

        info = predictions[name]

        p = out_dir / info["file"]

        if not p.exists():
            return False, (
                f"missing {name}"
            )

        if (
            sha_file(p)
            != info["sha256"]
        ):
            return False, (
                f"{name} checksum mismatch"
            )

        try:
            rows = (
                sum(
                    1
                    for _ in p.open(
                        encoding="utf-8"
                    )
                )
                - 1
            )

        except Exception:
            return False, (
                f"{name} unreadable"
            )

        if rows != info["rows"]:
            return False, (
                f"{name} row-count mismatch"
            )

    return True, "ok"


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--training-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--run-id",
        action="append",
    )

    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--preflight",
        action="store_true",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Frozen training chain
    # --------------------------------------------------------

    training_dir = (
        args.training_dir
    )

    data_dir = args.data_dir

    spec_path = (
        training_dir
        / "partC_training_spec.json"
    )

    source_path = (
        training_dir
        / "upstream"
        / "run_experiments.py"
    )

    config_path = (
        training_dir
        / "upstream"
        / "training_config.json"
    )

    spec = json.loads(
        spec_path.read_text(
            encoding="utf-8"
        )
    )

    implementation = spec[
        "training_implementation"
    ]

    bindings = spec[
        "dataset_binding"
    ]

    if (
        sha_file(source_path)
        != implementation[
            "source_runner_sha256"
        ]
    ):
        raise SystemExit(
            "[FATAL] source runner changed"
        )

    if (
        sha_file(config_path)
        != implementation[
            "source_training_config_sha256"
        ]
    ):
        raise SystemExit(
            "[FATAL] source config changed"
        )

    source = import_source(
        source_path
    )

    cfg, cfg_sha = (
        source.load_config(
            config_path
        )
    )

    if (
        cfg_sha
        != implementation[
            "source_training_config_sha256"
        ]
    ):
        raise SystemExit(
            "[FATAL] config loader "
            "hash disagreement"
        )

    if cfg["labels"] != EKMAN:
        raise SystemExit(
            "[FATAL] frozen label order "
            "changed"
        )

    if list(source.EKMAN) != EKMAN:
        raise SystemExit(
            "[FATAL] source label order "
            "changed"
        )

    # --------------------------------------------------------
    # Frozen V2 identity
    # --------------------------------------------------------

    manifest_path = (
        data_dir
        / "partC_dataset_manifest.json"
    )

    grid_path = (
        data_dir
        / "partC_run_grid.json"
    )

    freeze_path = (
        data_dir
        / "FREEZE.sha256"
    )

    identity = [
        (
            manifest_path,
            bindings[
                "partC_dataset_manifest_sha256"
            ],
            "dataset manifest",
        ),

        (
            grid_path,
            bindings[
                "partC_run_grid_sha256"
            ],
            "run grid",
        ),

        (
            freeze_path,
            bindings[
                "partC_dataset_freeze_sha256"
            ],
            "dataset freeze",
        ),
    ]

    for path, expected, name in identity:

        if (
            not path.exists()
            or sha_file(path) != expected
        ):
            raise SystemExit(
                f"[FATAL] {name} mismatch"
            )

    freeze_count = verify_freeze(
        data_dir
    )

    if freeze_count != 31:
        raise SystemExit(
            "[FATAL] expected 31 V2 "
            f"freeze entries, got {freeze_count}"
        )

    # --------------------------------------------------------
    # Fixed six-class metric proof
    # --------------------------------------------------------

    synthetic_y = np.asarray(
        [0, 1, 2, 3, 4]
    )

    synthetic_pred = np.asarray(
        [0, 1, 2, 3, 4]
    )

    metric, per_class = (
        source.macro_f1(
            synthetic_y,
            synthetic_pred,
        )
    )

    expected_metric = 5.0 / 6.0

    if (
        abs(
            metric
            - expected_metric
        )
        > 1e-12
    ):
        raise SystemExit(
            "[FATAL] macro-F1 is not "
            "fixed six-class"
        )

    if (
        len(per_class) != 6
        or per_class[5] != 0.0
    ):
        raise SystemExit(
            "[FATAL] absent sixth-class "
            "behavior changed"
        )

    twitter_val = pd.read_parquet(
        data_dir
        / "twitter_validation.parquet"
    )

    surprise_n = int(
        (
            twitter_val["label"]
            == "surprise"
        ).sum()
    )

    if surprise_n != 0:
        raise SystemExit(
            "[FATAL] Twitter validation "
            "support changed"
        )

    # --------------------------------------------------------
    # Frozen run grid
    # --------------------------------------------------------

    grid = json.loads(
        grid_path.read_text(
            encoding="utf-8"
        )
    )

    if len(grid) != 12:
        raise SystemExit(
            "[FATAL] expected 12 runs"
        )

    run_ids = [
        job["run_id"]
        for job in grid
    ]

    if (
        len(run_ids)
        != len(set(run_ids))
    ):
        raise SystemExit(
            "[FATAL] duplicate run-id"
        )

    for job in grid:

        for field in [
            "train_set",
            "native_validation",
        ]:

            p = (
                data_dir
                / job[field]
            )

            if not p.exists():
                raise SystemExit(
                    "[FATAL] missing "
                    f"{job[field]}"
                )

            df = pd.read_parquet(
                p
            )

            unknown = (
                set(
                    df["label"]
                    .astype(str)
                )
                - set(EKMAN)
            )

            if unknown:
                raise SystemExit(
                    "[FATAL] unknown "
                    f"labels: {unknown}"
                )

    print(
        f"[pre ] V2 freeze verified: "
        f"{freeze_count}/31"
    )

    print(
        "[pre ] fixed-six-class proof: "
        f"{metric:.12f} = 5/6"
    )

    print(
        "[pre ] sixth class in synthetic "
        f"proof: F1={per_class[5]}"
    )

    print(
        "[pre ] Twitter validation "
        f"surprise support={surprise_n}"
    )

    print(
        "[pre ] frozen grid: "
        "12 unique runs"
    )

    print(
        "[pre ] immutable source "
        "runner/config verified"
    )

    if args.preflight:

        print(
            "[pre ] PRE-FLIGHT PASSED"
        )

        print(
            "[pre ] no model loaded"
        )

        print(
            "[pre ] no model trained"
        )

        print(
            "[pre ] no test prediction "
            "generated"
        )

        return

    # --------------------------------------------------------
    # TRAINING — GPU only
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        raise SystemExit(
            "[FATAL] CUDA GPU required"
        )

    source.apply_determinism(
        cfg["determinism"]
    )

    if args.run_id:

        wanted = set(
            args.run_id
        )

        unknown = (
            wanted
            - set(run_ids)
        )

        if unknown:

            raise SystemExit(
                "[FATAL] unknown run-id(s): "
                f"{sorted(unknown)}"
            )

        selected = [
            job
            for job in grid
            if job["run_id"]
            in wanted
        ]

    else:
        selected = grid

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    this_runner_sha = sha_file(
        Path(__file__)
    )

    this_spec_sha = sha_file(
        spec_path
    )

    expected_completion = {
        "partC_runner_sha256":
            this_runner_sha,

        "partC_training_spec_sha256":
            this_spec_sha,

        "source_runner_sha256":
            sha_file(source_path),

        "source_training_config_sha256":
            sha_file(config_path),

        "partC_dataset_manifest_sha256":
            bindings[
                "partC_dataset_manifest_sha256"
            ],

        "partC_dataset_freeze_sha256":
            bindings[
                "partC_dataset_freeze_sha256"
            ],

        "partC_run_grid_sha256":
            bindings[
                "partC_run_grid_sha256"
            ],
    }

    reddit_val = pd.read_parquet(
        data_dir
        / "reddit_validation.parquet"
    )

    reddit_test = pd.read_parquet(
        data_dir
        / "reddit_test.parquet"
    )

    twitter_test = pd.read_parquet(
        data_dir
        / "twitter_test.parquet"
    )

    newly_completed = 0

    for job in selected:

        if (
            args.max_runs is not None
            and newly_completed
            >= args.max_runs
        ):
            break

        run_id = job["run_id"]

        ok, reason = complete(
            args.out_dir,
            run_id,
            expected_completion,
        )

        if ok:

            print(
                f"[skip] {run_id}: "
                "verified complete"
            )

            continue

        print(
            f"[run ] {run_id} "
            f"(previous={reason})"
        )

        train_path = (
            data_dir
            / job["train_set"]
        )

        native_val_path = (
            data_dir
            / job[
                "native_validation"
            ]
        )

        train_df = pd.read_parquet(
            train_path
        )

        native_val = (
            pd.read_parquet(
                native_val_path
            )
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # source.train_one uses native_val during training.
        # The concatenated evaluation pool is not touched
        # until the selected checkpoint has been restored.
        # ----------------------------------------------------

        evaluation_frames = [
            (
                "native_validation",
                native_val,
            ),

            (
                "reddit_validation",
                reddit_val.reset_index(
                    drop=True
                ),
            ),

            (
                "reddit_test",
                reddit_test.reset_index(
                    drop=True
                ),
            ),

            (
                "twitter_test",
                twitter_test.reset_index(
                    drop=True
                ),
            ),
        ]

        frames = []
        bounds = {}

        cursor = 0

        for name, frame in (
            evaluation_frames
        ):

            start = cursor
            stop = start + len(frame)

            bounds[name] = (
                start,
                stop,
            )

            frames.append(frame)

            cursor = stop

        eval_pool = pd.concat(
            frames,
            ignore_index=True,
        )

        print(
            f"[data] train="
            f"{len(train_df):,} "
            f"native-val="
            f"{len(native_val):,} "
            f"post-selection pool="
            f"{len(eval_pool):,}"
        )

        t0 = time.time()

        (
            probs,
            returned_y,
            native_val_f1,
            resolved_revision,
        ) = source.train_one(
            train_df,
            native_val,
            eval_pool,
            job["backbone"],
            int(job["seed"]),
            None,
            torch.device("cuda"),
            cfg,
        )

        elapsed = round(
            time.time() - t0,
            1,
        )

        expected_y = np.asarray([
            LABEL2ID[x]
            for x in eval_pool[
                "label"
            ].astype(str)
        ])

        if not np.array_equal(
            returned_y,
            expected_y,
        ):

            raise SystemExit(
                "[FATAL] returned labels "
                "do not match frozen "
                "evaluation pool"
            )

        paths = output_paths(
            args.out_dir,
            run_id,
        )

        output_meta = {}

        for name, frame in (
            evaluation_frames
        ):

            start, stop = (
                bounds[name]
            )

            subset_probs = (
                probs[start:stop]
            )

            pred = prediction_df(
                frame,
                subset_probs,
            )

            atomic_csv(
                paths[name],
                pred,
            )

            metrics = score(
                source,
                frame,
                subset_probs,
            )

            output_meta[name] = {
                "file":
                    paths[name].name,

                "rows":
                    int(len(pred)),

                "sha256":
                    sha_file(
                        paths[name]
                    ),

                "macro_f1":
                    metrics[
                        "macro_f1"
                    ],

                "per_class_f1":
                    metrics[
                        "per_class_f1"
                    ],
            }

        restored_native_f1 = (
            output_meta[
                "native_validation"
            ]["macro_f1"]
        )

        if (
            abs(
                float(native_val_f1)
                - restored_native_f1
            )
            > 1e-12
        ):

            raise SystemExit(
                "[FATAL] selected checkpoint "
                "native validation mismatch"
            )

        metadata = {
            "run_id":
                run_id,

            "backbone":
                job["backbone"],

            "model_seed":
                int(job["seed"]),

            "roles":
                job.get(
                    "roles",
                    [],
                ),

            "train_set":
                job["train_set"],

            "native_validation_set":
                job[
                    "native_validation"
                ],

            "n_train":
                int(len(train_df)),

            "n_native_validation":
                int(len(native_val)),

            "effective_train_ids_sha256":
                canonical_id_sha(
                    train_df
                ),

            "train_file_sha256":
                sha_file(
                    train_path
                ),

            "native_validation_file_sha256":
                sha_file(
                    native_val_path
                ),

            "validation_macro_f1":
                float(
                    native_val_f1
                ),

            "common_reddit_validation_macro_f1":
                output_meta[
                    "reddit_validation"
                ]["macro_f1"],

            "prediction_files":
                output_meta,

            "resolved_backbone_revision":
                str(
                    resolved_revision
                ),

            "partC_runner_sha256":
                this_runner_sha,

            "partC_training_spec_sha256":
                this_spec_sha,

            "source_runner_sha256":
                sha_file(
                    source_path
                ),

            "source_training_config_sha256":
                sha_file(
                    config_path
                ),

            "partC_dataset_manifest_sha256":
                bindings[
                    "partC_dataset_manifest_sha256"
                ],

            "partC_dataset_freeze_sha256":
                bindings[
                    "partC_dataset_freeze_sha256"
                ],

            "partC_run_grid_sha256":
                bindings[
                    "partC_run_grid_sha256"
                ],

            "sampling_correction_freeze_sha256":
                bindings[
                    "sampling_correction_freeze_sha256"
                ],

            "metric_semantics": {
                "label_space":
                    "fixed_Ekman_6",

                "undefined_class_f1":
                    0.0,
            },

            "twitter_test_timing": (
                "post-Part-B exploratory; "
                "gold outcomes known before "
                "Part-C execution"
            ),

            "elapsed_seconds":
                elapsed,

            "environment": {
                "python":
                    sys.version,

                "platform":
                    platform.platform(),

                "torch":
                    torch.__version__,

                "cuda_runtime":
                    torch.version.cuda,

                "gpu":
                    torch.cuda.get_device_name(
                        0
                    ),
            },
        }

        meta_path = (
            args.out_dir
            / f"{run_id}.meta.json"
        )

        atomic_text(
            meta_path,
            json.dumps(
                metadata,
                indent=2,
                sort_keys=True,
            ) + "\n",
        )

        ok, reason = complete(
            args.out_dir,
            run_id,
            expected_completion,
        )

        if not ok:

            raise SystemExit(
                "[FATAL] completion "
                f"verification failed: "
                f"{reason}"
            )

        newly_completed += 1

        print(
            f"[ok  ] {run_id} "
            f"native-val="
            f"{native_val_f1:.6f} "
            f"reddit-val="
            f"{output_meta['reddit_validation']['macro_f1']:.6f} "
            f"time={elapsed}s"
        )

    print(
        f"[done] newly completed: "
        f"{newly_completed}"
    )


if __name__ == "__main__":
    main()
