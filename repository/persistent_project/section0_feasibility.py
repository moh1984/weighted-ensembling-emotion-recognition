#!/usr/bin/env python3
"""
Protocol v6 - Section 0: pre-freeze feasibility check.

Uses corpus.py, so the corpus and the splits examined here are identical to
the ones build_dataset.py will produce with the same arguments. Trains nothing
and produces no performance number.

    python section0_feasibility.py --data-dir ./raw --ekman-map ./ekman_mapping.json

Add --skip-near-dedup only for a fast preview; the reported figures are then
NOT the ones the study will use and must not be used to freeze anything.
"""

import argparse
import json
from pathlib import Path

import corpus as C

MIN_ELIGIBLE_TEST = 1500
MIN_OVERLAP_PCT = 5.0
MIN_CLASS_SUPPORT = 50
CANDIDATE_DOSES = [1, 2, 3]


def probe_split(df, seed, doses=CANDIDATE_DOSES):
    d, eligible, elig_stats = C.build_r_clean(df, seed)

    n_train = int((d["role"] == "train").sum())
    n_val = int((d["role"] == "val").sum())
    n_test = int((d["role"] == "test").sum())
    n_res = int((d["role"] == "reserve").sum())
    analysis = max(n_train + n_val + n_test, 1)

    elig_test = d.index[(d["role"] == "test") & d["author"].isin(eligible)]
    support = (d.loc[elig_test, "label"].value_counts()
               .reindex(C.EKMAN).fillna(0).astype(int))

    res_per_author = d[d["role"] == "reserve"].groupby("author").size()

    # Dry-run each candidate dose AND the four-variable matched substitution,
    # so the freeze decision knows whether the matching actually holds up.
    dose_report = {}
    for k in doses:
        feasible_authors = int((res_per_author >= k).sum())
        coverage = feasible_authors / max(len(eligible), 1)
        ov, inj, rem, fb = C.build_r_overlap(d, eligible, k, seed)
        checks = C.verify_invariants(d, ov, inj, rem, eligible, k)
        denom = max(sum(fb[t] for t in C.MATCH_TAGS) + fb["failed"], 1)
        dose_report[k] = {
            "injected_ids_sha256": C.sha(d.loc[inj, "id"].tolist()),
            "removed_ids_sha256": C.sha(d.loc[rem, "id"].tolist()),
            "uniform_coverage_pct": round(100 * coverage, 2),
            "injected": len(inj),
            "overlap_pct_of_training": round(100 * len(inj) / max(n_train, 1), 3),
            "match_quality": fb,
            "match_level_pct": {t: round(100 * fb[t] / denom, 1)
                                for t in C.MATCH_TAGS},
            "exact_4var_match_pct": round(100 * fb["exact_4var"] / denom, 1),
            "label_only_pct": round(100 * fb["label_only"] / denom, 1),
            "invariants_ok": bool(all(checks.values())),
            "failed_invariants": [k2 for k2, v in checks.items() if not v],
        }

    split_hashes = {
        role: C.sha(d.index[d["role"] == role].tolist())
        for role in ("train", "val", "test", "reserve", "excluded")}
    split_hashes["eligible_test"] = C.sha(list(elig_test))

    return {
        "seed": seed,
        "split_hashes": split_hashes,
        "n_train": n_train, "n_val": n_val, "n_test": n_test, "n_reserve": n_res,
        "realized_ratios_pct": {
            "train": round(100 * n_train / analysis, 2),
            "val": round(100 * n_val / analysis, 2),
            "test": round(100 * n_test / analysis, 2)},
        "reserve_pct_of_corpus": round(100 * n_res / max(len(d), 1), 2),
        "n_test_authors": int(d[d["partition"] == "test"]["author"].nunique()),
        "n_eligible_authors": len(eligible),
        "eligibility": elig_stats,
        "n_excluded_same_thread": int((d["role"] == "excluded").sum()),
        "n_eligible_test_examples": int(len(elig_test)),
        "eligible_class_support": support.to_dict(),
        "min_eligible_class_support": int(support.min()) if len(elig_test) else 0,
        "reserve_per_author_median": float(res_per_author.median())
                                     if len(res_per_author) else 0.0,
        "dose_probe": dose_report,
    }


DISCORDANCE_ASSUMPTIONS = [0.05, 0.10, 0.20, 0.50]


def mde_paired(n, discordance, alpha_z=1.96, beta_z=0.84):
    """Minimum detectable accuracy difference for a paired (McNemar) test.

    Power for a paired binary comparison is driven by the number of DISCORDANT
    pairs, not by n alone, so a single MDE figure is meaningless without
    stating the assumed discordance rate. Under the normal approximation,
    detecting a difference needs |b - c| >= (z_alpha/2 + z_beta) * sqrt(b + c);
    with b + c = n * r this gives

        MDE = (z_alpha/2 + z_beta) * sqrt(r / n)

    expressed as an accuracy difference. Reported across a declared range of r
    rather than as one number.
    """
    if n <= 0 or discordance <= 0:
        return None
    return round(100 * (alpha_z + beta_z) * ((discordance / n) ** 0.5), 2)


def design_sensitivity(splits):
    """What could an H1 analysis on this corpus actually detect?"""
    out = {"assumed_discordance_rates": DISCORDANCE_ASSUMPTIONS,
           "note": ("Normal-approximation DESIGN SENSITIVITY, not post-hoc "
                    "achieved power. MDE is the smallest accuracy difference "
                    "detectable at 80% power, alpha = 0.05 two-sided, for a "
                    "paired test on the eligible test subset. It depends on "
                    "the discordance rate between the two systems, which is "
                    "unknown before the experiment, so a range of assumptions "
                    "is reported rather than a single figure. High discordance "
                    "means only large effects are detectable."),
           "per_split": {}}
    for sp in splits:
        n = sp["n_eligible_test_examples"]
        out["per_split"][sp["seed"]] = {
            "n_eligible_test": n,
            "mde_accuracy_points": {str(r): mde_paired(n, r)
                                    for r in DISCORDANCE_ASSUMPTIONS}}
    return out


def structural_diagnosis(corpus_stats, splits):
    """Is there an author-overlap MECHANISM in this corpus at all?

    Distinct from asking whether an effect was found. If almost every author
    contributes a single post, author overlap cannot occur at a meaningful
    rate, and a null result reflects the absence of a mechanism rather than
    the absence of a leakage effect in general.
    """
    reserve_pct = min(s["reserve_pct_of_corpus"] for s in splits)
    overlap_pct = min(s["dose_probe"][1]["overlap_pct_of_training"]
                      for s in splits) if splits else 0.0
    return {
        "posts_per_author_mean": round(
            corpus_stats["n_examples"] / max(corpus_stats["n_authors"], 1), 3),
        "pct_authors_ge2": corpus_stats["pct_authors_ge2"],
        "pct_posts_from_multipost_authors":
            corpus_stats["pct_posts_from_multipost_authors"],
        "min_reserve_pct_of_corpus": reserve_pct,
        "max_achievable_overlap_pct_of_training": overlap_pct,
        "mechanism_available": bool(overlap_pct >= MIN_OVERLAP_PCT),
        "interpretation": (
            "Author overlap is structurally near-absent in this corpus: the "
            "reserve pool - every post that could possibly be injected - is a "
            "small fraction of the corpus. A null stress-test result is "
            "therefore consistent with the absence of a mechanism and must "
            "NOT be read as evidence about the size of author leakage in "
            "emotion benchmarks generally."
            if overlap_pct < MIN_OVERLAP_PCT else
            "An author-overlap mechanism is available at the required dose."),
    }


def decide(splits):
    if any(s["n_eligible_test_examples"] == 0 for s in splits):
        return {"h1_confirmatory_feasible": False, "fixed_dose_k": None,
                "h1_primary_metric": None,
                "verdict": "NO ELIGIBLE TEST EXAMPLES. The author-level "
                           "design is not executable on this corpus."}

    min_elig = min(s["n_eligible_test_examples"] for s in splits)
    min_supp = min(s["min_eligible_class_support"] for s in splits)

    # Largest dose with full uniform coverage clearing the overlap floor in
    # EVERY split - the dose must be identical across splits.
    chosen = None
    for k in sorted(CANDIDATE_DOSES, reverse=True):
        if all(s["dose_probe"][k]["uniform_coverage_pct"] >= 99.9
               and s["dose_probe"][k]["overlap_pct_of_training"] >= MIN_OVERLAP_PCT
               and s["dose_probe"][k]["invariants_ok"] for s in splits):
            chosen = k
            break

    feasible = chosen is not None and min_elig >= MIN_ELIGIBLE_TEST
    metric = "macro_f1" if min_supp >= MIN_CLASS_SUPPORT else "accuracy"
    worst_exact = (min(s["dose_probe"][chosen]["exact_4var_match_pct"]
                       for s in splits) if chosen else None)
    worst_label_only = (max(s["dose_probe"][chosen]["label_only_pct"]
                            for s in splits) if chosen else None)

    # The dose that survives for an exploratory stress test is the largest
    # with full uniform coverage, even if it misses the confirmatory overlap
    # floor - coverage is a construction requirement, the floor is a power
    # requirement.
    explor = None
    for k in sorted(CANDIDATE_DOSES, reverse=True):
        if all(s["dose_probe"][k]["uniform_coverage_pct"] >= 99.9
               and s["dose_probe"][k]["invariants_ok"] for s in splits):
            explor = k
            break

    return {
        "h1_confirmatory_feasible": bool(feasible),
        "h1_status": "confirmatory" if feasible else "requires_amendment",
        "exploratory_dose_k": explor,
        "fixed_dose_k": chosen,
        "h1_primary_metric": metric,
        "min_eligible_test_examples": min_elig,
        "min_class_support_eligible": min_supp,
        "worst_exact_4var_match_pct": worst_exact,
        "worst_label_only_pct": worst_label_only,
        "matching_note": (
            "Hierarchical matching was fixed before model training. Training "
            "size and class composition are held exactly constant; subreddit, "
            "text length and author activity are matched hierarchically where "
            "feasible. The complete fallback distribution is reported and "
            "determines the strength of the descriptive claim - not whether "
            "the method is accepted post hoc."),
        "verdict": (
            f"PROCEED - freeze H1 as confirmatory with dose k={chosen} and "
            f"primary metric '{metric}'."
            if feasible else
            "DO NOT FREEZE H1 AS CONFIRMATORY. Choose explicitly, before any "
            "training: (a) demote H1 to exploratory, or (b) re-specify the "
            "grouping unit as link_id - which changes the construct from "
            "author leakage to thread leakage and requires the title, central "
            "message and H1 wording to change accordingly."),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--ekman-map", type=str, default=None)
    ap.add_argument("--out-prefix", type=str, default="section0_report")
    ap.add_argument("--skip-near-dedup", action="store_true")
    ap.add_argument("--allow-unpinned-dedup", action="store_true",
                    help="Permit an unpinned dedup model. Preview only; the resulting report cannot be used to freeze.")
    ap.add_argument("--allow-fallback-map", action="store_true",
                    help="Permit the embedded Ekman mapping. Preview only; "
                         "the resulting report must not be used to freeze.")
    args = ap.parse_args()

    if not (args.allow_unpinned_dedup or args.skip_near_dedup):
        C.require_pinned_dedup_model()
    elif args.skip_near_dedup:
        print("[gate ] --skip-near-dedup bypasses the near-duplicate pass and "
              "the pinned-model check; this report will be preview-only.")

    df, _, meta = C.build_corpus(args.data_dir, args.ekman_map,
                                 near_pass=not args.skip_near_dedup)

    # Section 0 decides the freeze, so it enforces the same provenance rule
    # as the builder: a freeze-quality report requires the official mapping.
    if meta["ekman_map_source"] != "official_file_verified" and not args.allow_fallback_map:
        raise SystemExit(
            "[FATAL] The official ekman_mapping.json was not supplied. This "
            "report would decide the freeze, so it must use the mapping file "
            "from the GoEmotions repository. Pass --ekman-map, or "
            "--allow-fallback-map for a preview that cannot be frozen.")

    ppa = df.groupby("author").size()
    corpus_stats = {
        "n_examples": int(len(df)),
        "n_authors": int(len(ppa)),
        "pct_authors_ge2": round(100 * float((ppa >= 2).mean()), 2),
        "pct_authors_ge3": round(100 * float((ppa >= 3).mean()), 2),
        "pct_authors_ge5": round(100 * float((ppa >= 5).mean()), 2),
        "pct_posts_from_multipost_authors":
            round(100 * float(ppa[ppa >= 2].sum() / ppa.sum()), 2),
        "max_posts_single_author": int(ppa.max()),
        "class_distribution": df["label"].value_counts().to_dict(),
    }
    print(f"\n[dist ] authors with >=2 posts: {corpus_stats['pct_authors_ge2']}% "
          f"| posts from them: {corpus_stats['pct_posts_from_multipost_authors']}%")

    splits = [probe_split(df, s) for s in C.SPLIT_SEEDS]
    verdict = decide(splits)

    L_agree = meta["agreement"]
    # A report is freeze-quality only if every provenance condition held.
    # The builder refuses any report that is not.
    freeze_quality = bool(
        meta["ekman_map_source"] == "official_file_verified"
        and meta["deduplication"]["near_pass_run"]
        and meta["deduplication"].get("model_revision_requested") is not None
        and meta["deduplication"].get("model_revision_verified")
        and not args.allow_fallback_map
        and not args.allow_unpinned_dedup
        and not args.skip_near_dedup)

    diagnosis = structural_diagnosis(corpus_stats, splits)
    sensitivity = design_sensitivity(splits)

    report = {"meta": meta, "corpus": corpus_stats, "splits": splits,
              "structural_diagnosis": diagnosis,
              "design_sensitivity": sensitivity,
              "freeze_quality": freeze_quality,
              "freeze_quality_conditions": {
                  "official_ekman_map": meta["ekman_map_source"] == "official_file_verified",
                  "near_dedup_ran": meta["deduplication"]["near_pass_run"],
                  "dedup_model_pinned":
                      meta["deduplication"].get("model_revision_requested") is not None,
                  "dedup_model_revision_verified":
                      bool(meta["deduplication"].get("model_revision_verified")),
                  "no_preview_overrides": not (args.allow_fallback_map
                                               or args.allow_unpinned_dedup
                                               or args.skip_near_dedup)},
              "decision": verdict,
              "confirmatory_split_seed": C.CONFIRMATORY_SEED,
              "thresholds": {"min_eligible_test": MIN_ELIGIBLE_TEST,
                             "min_overlap_pct": MIN_OVERLAP_PCT,
                             "min_class_support": MIN_CLASS_SUPPORT}}
    Path(f"{args.out_prefix}.json").write_text(json.dumps(report, indent=2))

    L = ["# Section 0 - Pre-freeze Feasibility Report", "",
         f"Ekman mapping source: **{meta['ekman_map_source']}**",
         f"Near-duplicate pass run: **{meta['deduplication']['near_pass_run']}**",
         f"Corpus content SHA-256: `{meta['corpus_content_sha256'][:16]}`",
         f"Corpus id-set SHA-256: `{meta['corpus_ids_sha256'][:16]}`",
         f"Ekman mapping SHA-256: `{str(meta['ekman_map_sha256'])[:16]}`",
         "", "Raw source files:"] + \
        [f"- {k}: `{str(v)[:16]}`" for k, v in meta["raw_file_sha256"].items()] + \
        ["", f"Near-duplicate algorithm: {meta['deduplication'].get('algorithm', 'n/a')}",
         f"Embedding model: `{meta['deduplication'].get('model')}` "
         f"(revision requested: {meta['deduplication'].get('model_revision_requested')}, "
         f"resolved: {meta['deduplication'].get('model_revision_resolved')})",
         "", "## Corpus", ""]
    for k, v in corpus_stats.items():
        if k != "class_distribution":
            L.append(f"- {k}: {v}")
    L += ["", "Class distribution: " + ", ".join(
        f"{k} {v:,}" for k, v in corpus_stats["class_distribution"].items()), ""]

    L += ["## Rater agreement (binary Fleiss' kappa per emotion)", ""]
    for e, v in L_agree["binary_per_emotion"].items():
        L.append(f"- {e}: kappa {v['kappa']} (n={v['n_items']:,})")
    L += ["", f"- rater annotations selecting >1 Ekman category: "
              f"{L_agree['pct_rater_annotations_multi_select']}%",
          f"- items containing at least one such annotation: "
          f"{L_agree['pct_items_with_any_multi_select']}%",
          f"- overall categorical kappa: omitted. "
          f"{L_agree['overall_omitted_because']}",
          "", "## Label construction by rater count", "",
          "| raters | items | single | multi | below threshold |",
          "|---|---|---|---|---|"]
    for n, v in meta["label_construction"]["by_rater_count"].items():
        L.append(f"| {n} | {v['items']:,} | {v['single_label']:,} | "
                 f"{v['multi_label']:,} | {v['below_agreement']:,} |")
    sens = meta["label_construction"]["sensitivity_strict_majority_rule"]
    L += ["", f"Sensitivity, strict-majority rule (`{sens['rule']}`): "
              f"single {sens['single_label']:,}, multi {sens['multi_label']:,}, "
              f"below {sens['below_threshold']:,}.",
          "A fixed >=2 threshold is a majority of three raters but only 40% of "
          "five, so items with more raters are likelier to reach the threshold "
          "in two categories and be routed to the multi-label subset."]

    L += ["", "## Structural diagnosis: is an author-overlap mechanism present?", ""]
    L += [f"- {k}: {v}" for k, v in diagnosis.items() if k != "interpretation"]
    L += ["", f"> {diagnosis['interpretation']}", "",
          "## Design sensitivity (MDE, 80% power, alpha = 0.05 two-sided)", "",
          "MDE depends on the discordance rate between the compared systems, "
          "which is unknown in advance, so a range of assumptions is reported "
          "rather than a single figure.", "",
          "| seed | eligible test N | " + " | ".join(
              f"MDE at r={r}" for r in DISCORDANCE_ASSUMPTIONS) + " |",
          "|---|---|" + "---|" * len(DISCORDANCE_ASSUMPTIONS)]
    for sd, v in sensitivity["per_split"].items():
        L.append(f"| {sd} | {v['n_eligible_test']:,} | " + " | ".join(
            f"{v['mde_accuracy_points'][str(r)]} pts"
            for r in DISCORDANCE_ASSUMPTIONS) + " |")

    L += ["", "## Splits", "",
          "| seed | train | val | test | reserve | excl. | ratios % | eligible "
          "(strict/loose) | eligible test | min class support |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for s in splits:
        r = s["realized_ratios_pct"]
        L.append(f"| {s['seed']} | {s['n_train']:,} | {s['n_val']:,} | "
                 f"{s['n_test']:,} | {s['n_reserve']:,} | "
                 f"{s['n_excluded_same_thread']:,} | "
                 f"{r['train']}/{r['val']}/{r['test']} | "
                 f"{s['eligibility']['strict_eligible']:,}/"
                 f"{s['eligibility']['loose_eligible']:,} | "
                 f"{s['n_eligible_test_examples']:,} | "
                 f"{s['min_eligible_class_support']} |")

    L += ["", "## Dose probe", "",
          "| seed | k | coverage % | injected | overlap % of train | "
          "exact 4-var % | -ppa % | -len % | label-only % | invariants |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for s in splits:
        for k, p in s["dose_probe"].items():
            L.append(f"| {s['seed']} | {k} | {p['uniform_coverage_pct']} | "
                     f"{p['injected']:,} | {p['overlap_pct_of_training']} | "
                     + " | ".join(str(p["match_level_pct"][t])
                                   for t in C.MATCH_TAGS) + " | "
                     f"{'OK' if p['invariants_ok'] else 'FAIL'} |")

    L += ["", f"**freeze_quality: {freeze_quality}**", ""]
    L += [f"- {k}: {v}" for k, v in report["freeze_quality_conditions"].items()]
    L += ["", "## Decision", ""] + [f"- **{k}**: {v}" for k, v in verdict.items()]
    Path(f"{args.out_prefix}.md").write_text("\n".join(L))

    print("\n" + "=" * 72)
    print(verdict["verdict"])
    print(f"freeze_quality: {freeze_quality}")
    if not freeze_quality:
        print("This report is PREVIEW ONLY and will be refused by "
              "build_dataset.py as authorization for a frozen build.")
    print("=" * 72)
    if not meta["deduplication"]["near_pass_run"]:
        print("[WARN ] near-duplicate pass did NOT run. These figures are a "
              "preview only and must not be used to freeze the protocol.")
    print(f"\nWrote {args.out_prefix}.md and {args.out_prefix}.json")


if __name__ == "__main__":
    main()
