#!/usr/bin/env python3

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


EKMAN = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
MATCH_TAGS = ["exact_4var", "no_ppa", "no_len", "label_only"]

N_PROVENANCE = 6
N_PER_SPLIT = 17
EXPECTED_TOTAL = N_PROVENANCE + 3 * N_PER_SPLIT


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_ids(ids):
    return hashlib.sha256(
        ",".join(sorted(map(str, ids))).encode("utf-8")
    ).hexdigest()


def read_ids(path):
    return [
        x.strip()
        for x in Path(path).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def pairwise_disjoint(sets):
    sets = list(sets)
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if sets[i] & sets[j]:
                return False
    return True


def class_counts(corpus, ids):
    sub = corpus.loc[list(ids)]
    return (
        sub["label"]
        .value_counts()
        .reindex(EKMAN, fill_value=0)
        .astype(int)
        .to_dict()
    )


def pct(x, n):
    return 100.0 * x / n if n else 0.0


class Checker:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def check(self, label, condition, detail=""):
        ok = bool(condition)
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {label}" + (f"  — {detail}" if detail else ""))
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        return ok


def build_match_index(train, rng):
    levels = [
        lambda r: (
            r["label"],
            r["subreddit"],
            r["len_bucket"],
            r["ppa_bucket"],
        ),
        lambda r: (
            r["label"],
            r["subreddit"],
            r["len_bucket"],
        ),
        lambda r: (
            r["label"],
            r["subreddit"],
        ),
        lambda r: (
            r["label"],
        ),
    ]

    idxs = [defaultdict(list) for _ in levels]

    for idx, r in train.iterrows():
        for lv, keyfn in enumerate(levels):
            idxs[lv][keyfn(r)].append(idx)

    for d in idxs:
        for k in d:
            rng.shuffle(d[k])

    return levels, idxs


def reconstruct_overlap(corpus, train_ids, reserve_ids, seed, dose_k):
    rng = np.random.default_rng(seed + 11)

    train = corpus[corpus.index.isin(train_ids)]
    reserve = corpus[corpus.index.isin(reserve_ids)]

    inject = []
    skipped = 0

    for author, grp in reserve.groupby("author"):
        if len(grp) >= dose_k:
            picked = grp.sample(
                n=dose_k,
                random_state=seed
            ).index.tolist()
            inject.extend(picked)
        else:
            skipped += 1

    levels, idxs = build_match_index(train, rng)

    used = set()
    removed = []
    fallback = {t: 0 for t in MATCH_TAGS}
    fallback["failed"] = 0

    for idx in inject:
        r = corpus.loc[idx]
        chosen = None

        for lv, keyfn in enumerate(levels):
            bucket = idxs[lv].get(keyfn(r))

            if not bucket:
                continue

            while bucket:
                cand = bucket.pop()

                if cand not in used:
                    used.add(cand)
                    fallback[MATCH_TAGS[lv]] += 1
                    chosen = cand
                    break

            if chosen is not None:
                break

        if chosen is None:
            fallback["failed"] += 1
        else:
            removed.append(chosen)

    fallback["authors_short_of_dose"] = skipped

    overlap = (set(train_ids) - set(removed)) | set(inject)

    return {
        "inject": set(map(str, inject)),
        "removed": set(map(str, removed)),
        "overlap": set(map(str, overlap)),
        "fallback": fallback,
    }


def strict_eligible_test_ids(corpus, test_ids):
    test_authors = set(corpus.loc[list(test_ids), "author"].astype(str))

    original_test_population = corpus[
        corpus["author"].astype(str).isin(test_authors)
    ]

    eligible_authors = set()

    for author, grp in original_test_population.groupby("author"):
        n_posts = len(grp)

        if "link_id_valid" in grp.columns:
            valid = grp[
                grp["link_id_valid"].fillna(False).astype(bool)
            ]
        else:
            valid = grp[grp["link_id"].notna()]

        n_threads = valid["link_id"].astype(str).nunique()

        if n_posts >= 2 and n_threads >= 2:
            eligible_authors.add(str(author))

    expected = {
        str(i)
        for i in test_ids
        if str(corpus.loc[str(i), "author"]) in eligible_authors
    }

    return expected, eligible_authors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", type=Path, required=True)
    ap.add_argument("--section0-report", type=Path, required=True)
    ap.add_argument("--h1-amendment", type=Path, required=True)
    args = ap.parse_args()

    root = args.splits_dir
    checker = Checker()

    dm = json.loads(
        (root / "dataset_manifest.json").read_text(encoding="utf-8")
    )

    s0 = json.loads(
        args.section0_report.read_text(encoding="utf-8")
    )

    amend = json.loads(
        args.h1_amendment.read_text(encoding="utf-8")
    )

    corpus = pd.read_parquet(root / "corpus.parquet")

    corpus["id"] = corpus["id"].astype(str)
    corpus["author"] = corpus["author"].astype(str)
    corpus = corpus.set_index("id", drop=False)

    all_ids = set(corpus.index)

    print("=" * 78)
    print("A. PROVENANCE / FREEZE CHAIN")
    print("=" * 78)

    section0_sha = sha256_file(args.section0_report)
    amendment_sha = sha256_file(args.h1_amendment)

    checker.check(
        "P01 Section-0 file SHA256 matches frozen manifest",
        section0_sha == dm.get("section0_report_sha256"),
    )

    checker.check(
        "P02 H1 amendment SHA256 matches frozen manifest",
        amendment_sha == dm.get("h1_amendment_sha256"),
    )

    checker.check(
        "P03 Amendment is bound to final Section-0",
        amend.get("section0_report_sha256") == section0_sha,
    )

    decision = s0.get("decision", {})

    checker.check(
        "P04 H1 status/dose/metric chain consistent",
        (
            s0.get("freeze_quality") is True
            and decision.get("h1_confirmatory_feasible") is False
            and amend.get("hypothesis") == "H1"
            and amend.get("to_status") == "exploratory"
            and dm.get("h1_status") == "exploratory"
            and int(amend.get("exploratory_dose_k")) == int(dm.get("dose_k"))
            and amend.get("h1_primary_metric") == "accuracy"
            and dm.get("h1_primary_metric") == "accuracy"
        ),
    )

    checker.check(
        "P05 Corpus content SHA chain consistent",
        (
            dm.get("corpus_content_sha256")
            == s0.get("meta", {}).get("corpus_content_sha256")
            == amend.get("corpus_content_sha256")
        ),
    )

    actual_dist = (
        corpus["label"]
        .value_counts()
        .reindex(EKMAN, fill_value=0)
        .astype(int)
        .to_dict()
    )

    expected_dist = {
        k: int(dm["corpus"]["class_distribution"].get(k, 0))
        for k in EKMAN
    }

    checker.check(
        "P06 Written corpus counts/authors/classes match frozen manifest",
        (
            len(corpus) == int(dm["corpus"]["n_examples"])
            and corpus["author"].nunique() == int(dm["corpus"]["n_authors"])
            and actual_dist == expected_dist
        ),
    )

    for seed in [42, 123, 2024]:
        print("\n" + "-" * 78)
        print(f"SPLIT {seed}")
        print("-" * 78)

        sd = root / f"seed{seed}"

        names = [
            "r_clean_train",
            "r_clean_val",
            "r_clean_test",
            "reserve",
            "excluded_same_thread",
            "r_overlap_train",
            "r_overlap_injected",
            "r_overlap_removed",
            "eligible_test",
            "r_random_train",
            "r_random_val",
            "r_random_test",
            "r_random_full_train",
            "r_random_full_val",
            "r_random_full_test",
        ]

        vals = {n: read_ids(sd / f"{n}.txt") for n in names}
        S = {n: set(v) for n, v in vals.items()}

        train = S["r_clean_train"]
        val = S["r_clean_val"]
        test = S["r_clean_test"]
        reserve = S["reserve"]
        excluded = S["excluded_same_thread"]
        overlap = S["r_overlap_train"]
        injected = S["r_overlap_injected"]
        removed = S["r_overlap_removed"]
        eligible = S["eligible_test"]

        checker.check(
            f"{seed}.01 IDs valid/no duplicates",
            set().union(*S.values()) <= all_ids
            and all(len(vals[n]) == len(S[n]) for n in names),
        )

        checker.check(
            f"{seed}.02 Core partitions pairwise disjoint",
            pairwise_disjoint([train, val, test, reserve, excluded]),
        )

        checker.check(
            f"{seed}.03 Core partitions cover corpus",
            set().union(train, val, test, reserve, excluded) == all_ids,
        )

        train_auth = set(corpus.loc[list(train), "author"])
        val_auth = set(corpus.loc[list(val), "author"])
        test_auth = set(corpus.loc[list(test), "author"])

        checker.check(
            f"{seed}.04 train/val author-disjoint",
            train_auth.isdisjoint(val_auth),
        )

        checker.check(
            f"{seed}.05 train/test author-disjoint",
            train_auth.isdisjoint(test_auth),
        )

        checker.check(
            f"{seed}.06 val/test author-disjoint",
            val_auth.isdisjoint(test_auth),
        )

        checker.check(
            f"{seed}.07 one test target per author",
            bool((corpus.loc[list(test), "author"].value_counts() == 1).all()),
        )

        expected_eligible, eligible_authors = strict_eligible_test_ids(
            corpus, test
        )

        checker.check(
            f"{seed}.08 eligible_test matches strict eligibility",
            eligible == expected_eligible,
        )

        checker.check(
            f"{seed}.09 injected from reserve / removed from train",
            injected <= reserve and removed <= train,
        )

        checker.check(
            f"{seed}.10 overlap exactly reconstructed",
            overlap == ((train - removed) | injected),
        )

        checker.check(
            f"{seed}.11 train sizes identical",
            len(train) == len(overlap),
        )

        checker.check(
            f"{seed}.12 class counts identical",
            class_counts(corpus, train) == class_counts(corpus, overlap),
        )

        dose_k = int(dm["dose_k"])

        inj_author_counts = (
            corpus.loc[list(injected), "author"]
            .astype(str)
            .value_counts()
        )

        eligible_author_set = set(
            corpus.loc[list(eligible), "author"].astype(str)
        )

        checker.check(
            f"{seed}.13 uniform coverage k={dose_k}",
            (
                set(inj_author_counts.index) == eligible_author_set
                and bool((inj_author_counts == dose_k).all())
            ),
        )

        target_thread = (
            corpus.loc[list(test)]
            .set_index("author")["link_id"]
            .astype(str)
            .to_dict()
        )

        thread_ok = all(
            str(corpus.loc[i, "link_id"])
            != str(target_thread.get(str(corpus.loc[i, "author"])))
            for i in injected
        )

        checker.check(
            f"{seed}.14 injected posts thread-disjoint",
            thread_ok,
        )

        recon = reconstruct_overlap(
            corpus,
            train,
            reserve,
            seed,
            dose_k,
        )

        checker.check(
            f"{seed}.15 matching reconstruction matches files",
            (
                recon["inject"] == injected
                and recon["removed"] == removed
                and recon["overlap"] == overlap
                and recon["fallback"]["failed"] == 0
            ),
        )

        checker.check(
            f"{seed}.16 random partitions valid",
            (
                pairwise_disjoint([
                    S["r_random_train"],
                    S["r_random_val"],
                    S["r_random_test"],
                ])
                and pairwise_disjoint([
                    S["r_random_full_train"],
                    S["r_random_full_val"],
                    S["r_random_full_test"],
                ])
            ),
        )

        jury = json.loads(
            (sd / "jury_blocks.json").read_text(encoding="utf-8")
        )

        blocks = {
            b: set(map(str, jury["blocks"][b]))
            for b in ["A", "B", "C"]
        }

        checker.check(
            f"{seed}.17 jury blocks partition train authors",
            (
                pairwise_disjoint(blocks.values())
                and set().union(*blocks.values())
                == set(map(str, train_auth))
            ),
        )

    total = checker.passed + checker.failed

    print("\n" + "=" * 78)
    print("FINAL")
    print("=" * 78)
    print("PASS:", checker.passed)
    print("FAIL:", checker.failed)
    print(f"TOTAL: {checker.passed}/{total} PASS")

    if total != EXPECTED_TOTAL:
        sys.exit(2)

    if checker.failed:
        print("\n❌ VERIFICATION FAILED")
        sys.exit(1)

    print("\n✅ 57/57 PASS — FROZEN SPLITS VERIFIED")


if __name__ == "__main__":
    main()
