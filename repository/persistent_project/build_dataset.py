#!/usr/bin/env python3
"""
Protocol v6 - build_dataset.py

Builds the frozen, hashed corpus and split artifacts for Part A (Reddit /
GoEmotions), using corpus.py so that construction is identical to what
section0_feasibility.py measured.

    python build_dataset.py --data-dir ./raw \
                            --ekman-map ./ekman_mapping.json \
                            --section0-report section0_report.json \
                            --out-dir ./splits

--dose-k exists only for preview builds; the frozen path takes the dose,
the H1 metric and every hash from the Section 0 report.

Fails hard - writing nothing - if any split invariant is violated, or if the
near-duplicate pass did not run (override with --allow-exact-only for a
preview build that must not be frozen).
"""

import argparse
import hashlib
import json
from datetime import datetime
import shutil
import tempfile
from pathlib import Path

import corpus as C


AMENDABLE = {"H1"}


def verify_h1_amendment(amend, section0, report_path, meta):
    """Accept a pre-registered demotion of H1, and nothing else.

    This is deliberately narrow. A general "downgrade a failing hypothesis"
    path would let any future feasibility failure be dissolved by relabelling,
    which is exactly the behaviour the pre-freeze machinery exists to prevent.
    The amendment therefore applies to H1 only, may only weaken it, and is
    bound by hash to the specific Section 0 report that triggered it.
    """
    h = amend.get("hypothesis")
    if h not in AMENDABLE:
        raise SystemExit(
            f"[FATAL] Amendments are permitted for {sorted(AMENDABLE)} only; "
            f"got {h!r}. H2 and H3 are confirmatory by construction and "
            "cannot be demoted by amendment - if either becomes infeasible, "
            "the protocol must be revised and re-frozen openly.")
    if amend.get("from_status") != "confirmatory" or \
       amend.get("to_status") != "exploratory":
        raise SystemExit(
            "[FATAL] An amendment may only weaken a hypothesis "
            "(confirmatory -> exploratory).")

    want = hashlib.sha256(Path(report_path).read_bytes()).hexdigest()
    if amend.get("section0_report_sha256") != want:
        raise SystemExit(
            "[FATAL] The amendment is not bound to this Section 0 report.\n"
            f"  amendment references : {amend.get('section0_report_sha256')}\n"
            f"  report on disk       : {want}\n"
            "An amendment must cite the exact report whose failure prompted "
            "it, so the order of events stays verifiable.")
    if amend.get("corpus_content_sha256") != meta["corpus_content_sha256"]:
        raise SystemExit("[FATAL] The amendment cites a different corpus.")

    for field in ("exploratory_dose_k", "h1_primary_metric",
                  "decision_text", "decided_by", "decided_at_utc"):
        if not amend.get(field):
            raise SystemExit(f"[FATAL] Amendment is missing '{field}'.")

    # A non-empty string is not a timestamp. The amendment's whole purpose is
    # to record WHEN the decision was taken relative to training, so the value
    # has to be a real instant, not "yesterday".
    ts = str(amend["decided_at_utc"])
    try:
        if not ts.endswith("Z"):
            raise ValueError("missing trailing Z")
        datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        raise SystemExit(
            f"[FATAL] decided_at_utc = {ts!r} is not an ISO-8601 UTC "
            "timestamp ending in Z (e.g. 2026-08-30T14:05:00Z).")

    # Unfilled template placeholders would leave the record unattributable.
    for field in ("decided_by", "decision_text"):
        val = str(amend[field])
        if val.startswith("<") or val.endswith(">") or len(val.strip()) < 3:
            raise SystemExit(
                f"[FATAL] Amendment field '{field}' still holds a template "
                f"placeholder ({val!r}). Fill it in before building.")

    explor = section0["decision"].get("exploratory_dose_k")
    if explor is not None and amend["exploratory_dose_k"] != explor:
        raise SystemExit(
            f"[FATAL] Amendment dose k={amend['exploratory_dose_k']} does not "
            f"match the dose Section 0 found viable (k={explor}).")

    # Declaring a metric is not the same as declaring the RIGHT one. Without
    # this check the amendment could substitute macro-F1 for the accuracy
    # Section 0 selected - reversing a pre-registered decision through the
    # very mechanism meant to record it.
    metric = section0["decision"].get("h1_primary_metric")
    if metric is not None and amend["h1_primary_metric"] != metric:
        raise SystemExit(
            f"[FATAL] Amendment metric {amend['h1_primary_metric']!r} does "
            f"not match the metric Section 0 selected ({metric!r}).")
    return "exploratory"


def check_split_hashes(seed, d, elig_test, inject, removed, dose_k, section0):
    """Every split partition must match what Section 0 measured."""
    if section0 is None:
        return {}
    rec = next((x for x in section0["splits"] if x["seed"] == seed), None)
    if rec is None:
        raise C.InvariantFailure(f"seed {seed} absent from the Section 0 report")
    got = {role: C.sha(d.index[d["role"] == role].tolist())
           for role in ("train", "val", "test", "reserve", "excluded")}
    got["eligible_test"] = C.sha(list(elig_test))
    for role, h in got.items():
        if rec["split_hashes"].get(role) != h:
            raise C.InvariantFailure(
                f"seed {seed}: '{role}' partition differs from the Section 0 "
                f"report. The frozen build is not the analysed split.")
    probe = rec["dose_probe"].get(str(dose_k)) or rec["dose_probe"].get(dose_k)
    if probe:
        pairs = (("injected_ids_sha256", inject), ("removed_ids_sha256", removed))
        for key, ids in pairs:
            if probe.get(key) and probe[key] != C.sha(d.loc[ids, "id"].tolist()):
                raise C.InvariantFailure(
                    f"seed {seed}: {key} differs from the Section 0 report; "
                    "the substitution matching is not reproducing.")
    return got


def build_split_artifacts(df, seed, dose_k, section0=None):
    """Compute everything for one split instance. Writes nothing."""
    d, eligible, elig_stats = C.build_r_clean(df, seed)

    clean = {r: d.index[d["role"] == r].tolist()
             for r in ("train", "val", "test", "reserve", "excluded")}
    ov_train, inject, removed, fallback = C.build_r_overlap(
        d, eligible, dose_k, seed)
    analysis_pop = d[d["role"].isin(["train", "val", "test"])]
    rand = C.build_r_random(analysis_pop, seed)          # size-matched to R_clean
    rand_full = C.build_r_random(d, seed)                # literature comparability
    blocks, block_stats = C.complementary_blocks(d, seed)

    elig_test = [i for i in clean["test"] if d.loc[i, "author"] in eligible]
    checks = C.verify_invariants(d, ov_train, inject, removed,
                                 eligible, dose_k)

    check_split_hashes(seed, d, elig_test, inject, removed, dose_k, section0)

    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise C.InvariantFailure(
            f"seed {seed}: split invariants violated: {failed}. "
            f"No artifacts written.")
    if fallback["failed"]:
        raise C.InvariantFailure(
            f"seed {seed}: {fallback['failed']} injections had no substitution "
            f"partner, so training size cannot be held constant. "
            f"No artifacts written.")

    files = {
        "r_clean_train.txt": clean["train"],
        "r_clean_val.txt": clean["val"],
        "r_clean_test.txt": clean["test"],
        "reserve.txt": clean["reserve"],
        "excluded_same_thread.txt": clean["excluded"],
        "r_overlap_train.txt": ov_train,
        "r_overlap_injected.txt": inject,
        "r_overlap_removed.txt": removed,
        "eligible_test.txt": elig_test,
        "r_random_train.txt": rand["train"],
        "r_random_val.txt": rand["val"],
        "r_random_test.txt": rand["test"],
        "r_random_full_train.txt": rand_full["train"],
        "r_random_full_val.txt": rand_full["val"],
        "r_random_full_test.txt": rand_full["test"],
    }

    analysis = max(len(clean["train"]) + len(clean["val"]) + len(clean["test"]), 1)
    support = (d.loc[elig_test, "label"].value_counts()
               .reindex(C.EKMAN).fillna(0).astype(int))
    denom = max(sum(fallback[t] for t in C.MATCH_TAGS), 1)

    manifest = {
        "seed": seed,
        "is_confirmatory_split": seed == C.CONFIRMATORY_SEED,
        "dose_k": dose_k,
        "counts": {k: len(v) for k, v in clean.items()} | {
            "overlap_train": len(ov_train), "injected": len(inject),
            "removed": len(removed), "eligible_test": len(elig_test)},
        "realized_ratios_pct": {
            "train": round(100 * len(clean["train"]) / analysis, 2),
            "val": round(100 * len(clean["val"]) / analysis, 2),
            "test": round(100 * len(clean["test"]) / analysis, 2)},
        "reserve_pct_of_corpus": round(100 * len(clean["reserve"]) / len(d), 2),
        "overlap_pct_of_training":
            round(100 * len(inject) / max(len(clean["train"]), 1), 3),
        "n_eligible_authors": len(eligible),
        "eligible_class_support": support.to_dict(),
        "min_eligible_class_support": int(support.min()) if elig_test else 0,
        "substitution_match_quality": fallback,
        "exact_4var_match_pct": round(100 * fallback["exact_4var"] / denom, 1),
        "label_only_pct": round(100 * fallback["label_only"] / denom, 1),
        "eligibility": elig_stats,
        "r_random_population": ("r_random: analysis population (train+val+test), "
                                "size-matched to R_clean; r_random_full: whole "
                                "cleaned corpus, for literature comparability"),
        "jury_blocks": block_stats,
        "invariant_checks": checks,
        "hashes": {name: C.sha(vals) for name, vals in files.items()},
    }
    jury = {"blocks": blocks, "jurors": {"J1": ["A", "B"], "J2": ["A", "C"],
                                         "J3": ["B", "C"]}, **block_stats}
    return files, jury, manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--ekman-map", type=str, default=None)
    ap.add_argument("--section0-report", type=Path, default=None,
                    help="section0_report.json. Supplies dose_k and is checked "
                         "against the rebuilt corpus hash. Required for a "
                         "frozen build.")
    ap.add_argument("--dose-k", type=int, default=None,
                    help="Manual override for preview builds only. A frozen "
                         "build must take the dose from --section0-report.")
    ap.add_argument("--out-dir", type=Path, default=Path("splits"))
    ap.add_argument("--h1-amendment", type=Path, default=None,
                    help="Signed H1 demotion bound to the Section 0 report. "
                         "Required when Section 0 declined H1 as confirmatory.")
    ap.add_argument("--allow-unpinned-dedup", action="store_true",
                    help="Permit an unpinned dedup model. Preview only.")
    ap.add_argument("--allow-fallback-map", action="store_true",
                    help="Permit the embedded Ekman mapping. Preview only.")
    ap.add_argument("--allow-exact-only", action="store_true",
                    help="Permit a preview build without the near-duplicate "
                         "pass. Such a build must not be frozen.")
    args = ap.parse_args()

    if not args.allow_unpinned_dedup:
        C.require_pinned_dedup_model()

    if args.section0_report is None and args.dose_k is None:
        raise SystemExit("[FATAL] Supply --section0-report (frozen build) or "
                         "--dose-k (preview build).")

    df, multi_ref, meta = C.build_corpus(args.data_dir, args.ekman_map,
                                         near_pass=True)

    section0 = None
    if args.section0_report is not None:
        section0 = json.loads(Path(args.section0_report).read_text())

        # A frozen build must not be constructible out of preview parts. The
        # report has to be freeze-quality, and no preview override may be
        # combined with it - otherwise "frozen" and "preview" differ only by a
        # field in a manifest that nobody reads.
        if not section0.get("freeze_quality", False):
            conds = section0.get("freeze_quality_conditions", {})
            failed = [k for k, v in conds.items() if not v] or ["unknown"]
            raise SystemExit(
                "[FATAL] The Section 0 report is preview-only and cannot "
                f"authorize a frozen build. Unmet conditions: {failed}. "
                "Re-run Section 0 with the official mapping, the "
                "near-duplicate pass enabled and the dedup model revision "
                "pinned.")
        overrides = [n for n, v in (("--allow-unpinned-dedup", args.allow_unpinned_dedup),
                                    ("--allow-fallback-map", args.allow_fallback_map),
                                    ("--allow-exact-only", args.allow_exact_only))
                     if v]
        if overrides:
            raise SystemExit(
                f"[FATAL] Preview overrides {overrides} cannot be combined "
                "with --section0-report. A frozen build takes no shortcuts.")
        # The whole protocol rests on the dose being fixed from Section 0
        # before any result is seen. Passing it by hand leaves that guarantee
        # to convention; checking the corpus hash makes a frozen build that
        # differs from the feasibility analysis impossible.
        # Section 0 does not merely choose a dose - it grants methodological
        # permission to build H1 as a confirmatory hypothesis. A report that
        # names a technically workable dose while declaring H1 infeasible must
        # not be usable as a licence to proceed.
        h1_status = "confirmatory"
        if not section0["decision"].get("h1_confirmatory_feasible"):
            if args.h1_amendment is None:
                raise SystemExit(
                    "[FATAL] Section 0 did not approve H1 as confirmatory "
                    f"(verdict: {section0['decision'].get('verdict', 'n/a')}). "
                    "Proceeding requires a signed H1 amendment file bound to "
                    "this report; pass --h1-amendment. Editing the report by "
                    "hand is not a substitute.")
            amend = json.loads(Path(args.h1_amendment).read_text())
            h1_status = verify_h1_amendment(amend, section0,
                                            args.section0_report, meta)
            print(f"[amend] H1 demoted to '{h1_status}' by amendment dated "
                  f"{amend.get('decided_at_utc')} "
                  f"(dose k={amend.get('exploratory_dose_k')}, "
                  f"metric '{amend.get('h1_primary_metric')}')")

        for key, label in (("corpus_content_sha256", "corpus content"),
                           ("corpus_ids_sha256", "corpus id set"),
                           ("ekman_map_sha256", "Ekman mapping file"),
                           ("raw_file_sha256", "raw GoEmotions files")):
            want, got = section0["meta"].get(key), meta.get(key)
            if want != got:
                raise SystemExit(
                    f"[FATAL] {label} hash mismatch.\n"
                    f"  Section 0 report : {want}\n"
                    f"  this build       : {got}\n"
                    "The feasibility analysis and this build are not the same "
                    "inputs. Re-run Section 0 with identical arguments.")
        dose_from_report = (section0["decision"]["fixed_dose_k"]
                            if h1_status == "confirmatory"
                            else amend["exploratory_dose_k"])
        if dose_from_report is None:
            raise SystemExit(
                "[FATAL] The Section 0 report records no feasible dose "
                "(fixed_dose_k is null). H1 cannot be built as specified; the "
                "authors must decide between demoting H1 to exploratory and "
                "re-specifying the grouping unit, before any training.")
        if args.dose_k is not None and args.dose_k != dose_from_report:
            raise SystemExit(
                f"[FATAL] --dose-k {args.dose_k} contradicts the Section 0 "
                f"report (k={dose_from_report}). Selecting a dose by hand "
                "defeats the pre-registration.")
        args.dose_k = dose_from_report
        print(f"[sec0 ] content + id + mapping hashes verified; "
              f"dose k={args.dose_k}, "
              f"H1 metric '{section0['decision']['h1_primary_metric']}'")

    if meta["ekman_map_source"] != "official_file_verified" and not args.allow_fallback_map:
        raise SystemExit(
            "[FATAL] The official ekman_mapping.json was not supplied. A "
            "frozen build must use the mapping file from the GoEmotions "
            "repository, not the embedded copy. Pass --ekman-map, or "
            "--allow-fallback-map for a preview that will not be frozen.")

    if not meta["deduplication"]["near_pass_run"] and not args.allow_exact_only:
        raise SystemExit(
            "[FATAL] The near-duplicate pass did not run (install "
            "sentence-transformers and scikit-learn). Without it, R_clean and "
            "R_overlap would differ in duplicate handling as well as author "
            "overlap, confounding H1. Re-run with --allow-exact-only only for "
            "a preview that will not be frozen.")

    # Compute everything first; write only if every split passes.
    results = []
    for seed in C.SPLIT_SEEDS:
        print(f"[split] seed {seed} ...")
        results.append((seed,) + build_split_artifacts(df, seed, args.dose_k,
                                                       section0))
        m = results[-1][3]
        print(f"[seed{seed}] train {m['counts']['train']:,} | reserve "
              f"{m['counts']['reserve']:,} | injected {m['counts']['injected']:,} "
              f"({m['overlap_pct_of_training']}% of train) | eligible test "
              f"{m['counts']['eligible_test']:,} | ratios "
              f"{m['realized_ratios_pct']} | invariants OK")

    # Atomic write: stage in a temp dir, then swap into place.
    tmp = Path(tempfile.mkdtemp(prefix="splits_", dir=args.out_dir.parent))
    try:
        fmt = C.write_table(df, tmp / "corpus")
        C.write_table(multi_ref, tmp / "multilabel_reference")
        manifests = []
        for seed, files, jury, manifest in results:
            sd = tmp / f"seed{seed}"
            sd.mkdir(parents=True)
            for name, vals in files.items():
                (sd / name).write_text("\n".join(map(str, vals)))
            (sd / "jury_blocks.json").write_text(json.dumps(jury, indent=2))
            (sd / "manifest.json").write_text(json.dumps(manifest, indent=2))
            manifests.append(manifest)

        ppa = df.groupby("author").size()
        (tmp / "agreement_report.json").write_text(
            json.dumps(meta["agreement"], indent=2))
        (tmp / "dataset_manifest.json").write_text(json.dumps({
            **meta,
            "table_format": fmt,
            "dose_k": args.dose_k,
            "dose_source": ("section0_report" if section0 else "manual_override"),
            "h1_primary_metric": (section0["decision"]["h1_primary_metric"]
                                  if section0 else None),
            "h1_status": (h1_status if section0 else None),
            "section0_report_sha256": (
                hashlib.sha256(Path(args.section0_report).read_bytes()).hexdigest()
                if args.section0_report else None),
            "h1_amendment_decided_at_utc": (
                json.loads(Path(args.h1_amendment).read_text()).get("decided_at_utc")
                if args.h1_amendment else None),
            "h1_amendment_sha256": (
                hashlib.sha256(Path(args.h1_amendment).read_bytes()).hexdigest()
                if args.h1_amendment else None),
            "freeze_quality": bool(
                section0 is not None
                and meta["ekman_map_source"] == "official_file_verified"
                and meta["deduplication"]["near_pass_run"]
                and meta["deduplication"].get("model_revision_requested")
                is not None
                and meta["deduplication"].get("model_revision_verified")),
            "confirmatory_split_seed": C.CONFIRMATORY_SEED,
            "corpus": {"n_examples": int(len(df)), "n_authors": int(len(ppa)),
                       "class_distribution": df["label"].value_counts().to_dict()},
            "splits": manifests,
        }, indent=2))

        if args.out_dir.exists():
            shutil.rmtree(args.out_dir)
        tmp.rename(args.out_dir)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    print(f"\n[done ] corpus {len(df):,} examples, "
          f"{df['author'].nunique():,} authors")
    print(f"[done ] wrote {args.out_dir}/dataset_manifest.json")


if __name__ == "__main__":
    main()
