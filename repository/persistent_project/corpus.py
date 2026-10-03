#!/usr/bin/env python3
"""
Protocol v6 - corpus.py

Single source of truth for corpus construction, agreement statistics, and
split assignment. Both section0_feasibility.py and build_dataset.py import
from here, so the corpus the feasibility check measures is the corpus the
builder produces.
"""

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- constants

SPLIT_SEEDS = [42, 123, 2024]
CONFIRMATORY_SEED = 42
TARGET_TEST = 0.15
TARGET_VAL = 0.15
EKMAN = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
# Ekman-level consensus threshold, inspired by (but not identical to)
# GoEmotions' own >=2-rater filtering, which is applied to the 27
# fine-grained labels. We apply it AFTER mapping to Ekman-6.
MIN_RATER_AGREEMENT = 2
DEDUP_THRESHOLD = 0.85
DEDUP_NEIGHBORS = 10
DEDUP_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Pin a commit here once chosen. The embedding model is part of DATASET
# CONSTRUCTION, not of training: if it changes, the surviving corpus changes,
# the content fingerprint changes, and every hash check in the freeze chain
# fails. Leaving it unpinned makes the freeze guarantee conditional on an
# external repository not being updated.
DEDUP_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
INVALID_AUTHORS = {"[deleted]", "[removed]", "", "none", "nan", "automoderator"}
RESERVE_WARN_FRACTION = 0.25

FALLBACK_EKMAN_MAP = {
    "anger":    ["anger", "annoyance", "disapproval"],
    "disgust":  ["disgust"],
    "fear":     ["fear", "nervousness"],
    "joy":      ["joy", "amusement", "approval", "excitement", "gratitude",
                 "love", "optimism", "relief", "pride", "admiration",
                 "desire", "caring"],
    "sadness":  ["sadness", "disappointment", "embarrassment", "grief",
                 "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
}


class InvariantFailure(RuntimeError):
    pass


FINGERPRINT_COLS = ["id", "text", "label", "author", "subreddit", "link_id",
                    "created_utc_int", "len_bucket", "ppa_bucket"]


def sha(items):
    return hashlib.sha256(",".join(map(str, sorted(items))).encode()).hexdigest()


def sha_file(path):
    if path is None or not Path(path).exists():
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def corpus_fingerprint(corpus):
    """Hash the corpus CONTENT, not just its identifiers.

    An id-list hash is not a fingerprint of the study: the Ekman mapping,
    label assignment, author cleaning, bucket definitions and even the text
    could all change while the surviving id set stayed the same. Hashing the
    canonical columns catches those; the id hash is kept alongside as a
    cheaper diagnostic that localises WHERE a mismatch arose.
    """
    cols = [c for c in FINGERPRINT_COLS if c in corpus.columns]
    # corpus carries 'id' as both index and column, so drop the index level
    # before sorting to avoid an ambiguous label.
    payload = (corpus[cols].reset_index(drop=True).sort_values("id")
               .astype(str).to_csv(index=False, lineterminator="\n"))
    return hashlib.sha256(payload.encode()).hexdigest()


# ------------------------------------------------------- corpus construction

def load_raw(data_dir: Path):
    """Load the three raw files and hash each one.

    The processed corpus could in principle be reproduced from slightly
    different raw annotation files - with different agreement statistics -
    so the raw hashes tie the whole chain to the actual source.
    """
    frames, hashes = [], {}
    for i in (1, 2, 3):
        p = Path(data_dir) / f"goemotions_{i}.csv"
        if not p.exists():
            raise FileNotFoundError(p)
        hashes[p.name] = sha_file(p)
        frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    print(f"[load ] {len(df):,} rater rows from {len(hashes)} files")
    return df, hashes


def _canonical_map(m):
    return {k: sorted(v) for k, v in sorted(m.items())}


def load_ekman_map(path):
    """Load the mapping and verify its CONTENT against the reference.

    Treating "a file exists at this path" as proof of provenance is not
    enough: an edited mapping would be loaded by Section 0 and by the builder
    alike, so every downstream hash would agree and the whole chain would
    certify a mapping nobody verified. The file's content is therefore
    compared against the reference copy, and only then called verified. The
    file hash is still recorded, so the exact bytes remain identifiable.
    """
    if path and Path(path).exists():
        mapping = json.load(open(path))
        if _canonical_map(mapping) != _canonical_map(FALLBACK_EKMAN_MAP):
            raise SystemExit(
                f"[FATAL] The Ekman mapping in {path} does not match the "
                "verified GoEmotions mapping. Either the file was edited or "
                "the upstream mapping changed; in both cases a frozen build "
                "must not proceed silently. Compare against "
                "google-research/goemotions/data/ekman_mapping.json.")
        return mapping, "official_file_verified"
    print("[map  ] WARNING: embedded fallback mapping in use.")
    return FALLBACK_EKMAN_MAP, "embedded_fallback"


def rater_level_ekman(df, ekman_map):
    """Map each RATER's annotation to Ekman-6 before any aggregation.

    A rater may select several fine-grained labels (e.g. joy AND amusement)
    that fold into the same Ekman category. Summing the fine-grained columns
    across raters first - as an earlier version did - would count that single
    rater twice. Mapping per rater and clipping to {0,1} gives each rater at
    most one vote per Ekman category, which is what the aggregation rule
    below assumes.
    """
    fine = sorted({l for v in ekman_map.values() for l in v})
    missing = [c for c in fine if c not in df.columns]
    if missing:
        raise ValueError(f"mapping refers to absent columns: {missing}")

    if "example_very_unclear" in df.columns:
        n0 = len(df)
        df = df[~df["example_very_unclear"].fillna(False).astype(bool)]
        print(f"[filt ] dropped {n0 - len(df):,} 'very unclear' rater rows")

    out = pd.DataFrame({"id": df["id"].to_numpy()})
    for coarse in EKMAN:
        members = [m for m in ekman_map[coarse] if m in df.columns]
        out[coarse] = (df[members].sum(axis=1).to_numpy() > 0).astype(int)
    if "rater_id" in df.columns:
        out["rater_id"] = df["rater_id"].to_numpy()

    # created_utc is carried even though Phase 1 does not use it: the
    # protocol names it as a reason for using the raw files and Phase 3
    # needs it. Freezing the corpus without it would force a rebuild and
    # invalidate every hash in the chain.
    meta_cols = [c for c in ("text", "author", "subreddit", "link_id",
                             "created_utc") if c in df.columns]
    meta = df.drop_duplicates("id").set_index("id")[meta_cols]
    return out, meta


def aggregate_to_examples(rater_ek, meta, min_agreement=MIN_RATER_AGREEMENT):
    """Rater-level Ekman votes -> per-example labels.

    Primary (single-label) corpus: exactly ONE Ekman category reaches
    `min_agreement` raters. Two or more categories reaching it go to the
    multi-label subset; none reaching it is dropped as low-agreement.

    An 'any vote counts' rule would admit examples supported by a single
    rater while discarding clear 2-vs-1 majorities as multi-label - exactly
    backwards.
    """
    votes = rater_ek.groupby("id")[EKMAN].sum()
    n_raters = rater_ek.groupby("id").size().rename("n_raters")

    reached = (votes >= min_agreement)
    n_reached = reached.sum(axis=1)

    out = votes.join(n_raters).join(meta).reset_index()
    out["n_reached"] = n_reached.to_numpy()
    # label = the category that reached agreement (ties impossible when n_reached==1)
    out["label"] = votes.idxmax(axis=1).to_numpy()
    out["top_votes"] = votes.max(axis=1).to_numpy()

    single = out[out["n_reached"] == 1].copy()
    multi = out[out["n_reached"] > 1].copy()
    dropped = int((out["n_reached"] == 0).sum())

    # A fixed >=2 threshold means different agreement PROPORTIONS depending
    # on how many raters an item received: 2/3 is a majority, 2/5 is not.
    # Items with more raters are therefore likelier to have two categories
    # reach the threshold and be pushed into the multi-label subset. Report
    # the breakdown so this selection effect is visible rather than hidden,
    # and provide a proportional-rule sensitivity count.
    by_n = {}
    n_map = n_raters.reindex(out["id"].to_numpy()).to_numpy()
    for n in sorted(set(n_map.tolist())):
        m = n_map == n
        by_n[int(n)] = {
            "items": int(m.sum()),
            "single_label": int(((out["n_reached"] == 1).to_numpy() & m).sum()),
            "multi_label": int(((out["n_reached"] > 1).to_numpy() & m).sum()),
            "below_agreement": int(((out["n_reached"] == 0).to_numpy() & m).sum()),
        }

    strict_maj = (votes.to_numpy() > (n_raters.reindex(votes.index)
                                      .to_numpy()[:, None] / 2))
    n_strict = strict_maj.sum(axis=1)

    stats = {
        "examples_total": int(len(out)),
        "min_agreement": min_agreement,
        "single_label": int(len(single)),
        "multi_label": int(len(multi)),
        "below_agreement_dropped": dropped,
        "rater_count_distribution":
            {int(k): int(v) for k, v in n_raters.value_counts().items()},
        "by_rater_count": by_n,
        "sensitivity_strict_majority_rule": {
            "rule": "votes > n_raters / 2",
            "single_label": int((n_strict == 1).sum()),
            "multi_label": int((n_strict > 1).sum()),
            "below_threshold": int((n_strict == 0).sum()),
        },
    }
    print(f"[label] single {len(single):,} | multi {len(multi):,} | "
          f"below-agreement {dropped:,}  (rule: >={min_agreement} raters)")
    return single, multi, stats


# ------------------------------------------------------ rater agreement (κ)

def _fleiss_kappa_variable(counts):
    """Fleiss' kappa allowing a different number of raters per item.

    counts: (n_items, n_categories) integer array of rater votes per item.
    Items with fewer than two raters are excluded (undefined agreement).
    """
    counts = np.asarray(counts, dtype=float)
    n_i = counts.sum(axis=1)
    keep = n_i >= 2
    counts, n_i = counts[keep], n_i[keep]
    if len(counts) == 0:
        return float("nan"), 0
    P_i = (np.sum(counts * (counts - 1), axis=1)) / (n_i * (n_i - 1))
    P_bar = P_i.mean()
    p_j = counts.sum(axis=0) / n_i.sum()
    P_e = np.sum(p_j ** 2)
    kappa = (P_bar - P_e) / (1 - P_e) if (1 - P_e) > 0 else float("nan")
    return float(kappa), int(len(counts))


def compute_agreement(rater_ek):
    """Binary Fleiss' kappa per Ekman emotion. Six figures, no overall kappa.

    An overall CATEGORICAL Fleiss' kappa is deliberately NOT reported.
    GoEmotions raters may select several emotions for one item, so the votes
    for an item can sum to more than the number of raters (a rater choosing
    joy and surprise contributes to two Ekman categories). Categorical
    Fleiss' kappa assumes each rater assigns an item to exactly one category,
    i.e. that the row sums equal the rater count, and that assumption is
    violated here - the resulting statistic would not mean what its name says.

    The per-emotion binary form has no such problem: each rater contributes
    exactly one yes/no verdict per emotion, so row sums equal the rater count
    by construction.

    If an aggregate figure is ever wanted, use Krippendorff's alpha with MASI
    distance, which is defined for multi-label annotation.
    """
    grouped = rater_ek.groupby("id")[EKMAN].sum()
    n_raters = rater_ek.groupby("id").size()

    per_emotion = {}
    for e in EKMAN:
        pos = grouped[e].to_numpy()
        counts = np.stack([pos, n_raters.to_numpy() - pos], axis=1)
        k, n = _fleiss_kappa_variable(counts)
        per_emotion[e] = {"kappa": round(k, 4), "n_items": n}

    # Measured on each RATER ROW, not on per-item sums. An item-level
    # comparison of total votes against the rater count cancels out when one
    # rater multi-selects and another abstains, so it under-reports.
    per_row = rater_ek[EKMAN].sum(axis=1).to_numpy()
    pct_rows = float((per_row > 1).mean())
    item_has_multi = (rater_ek.assign(_n=per_row)
                      .groupby("id")["_n"].max() > 1)
    return {
        "method": ("Binary Fleiss-type kappa per emotion, with a variable "
                   "number of raters per item. Not classical Fleiss' kappa, "
                   "which assumes a fixed rater count. Item-level agreement is "
                   "averaged equally across items (Pbar), while marginal "
                   "category frequencies (p_j) are estimated from the pooled "
                   "ratings and therefore reflect the differing numbers of "
                   "ratings per item."),
        "binary_per_emotion": per_emotion,
        "overall_categorical_kappa": None,
        "overall_omitted_because": (
            "Raters may select multiple emotions, so per-item votes can exceed "
            "the rater count; categorical Fleiss' kappa is undefined under "
            "multi-label annotation. Krippendorff's alpha with MASI distance "
            "is the defined alternative if an aggregate figure is wanted."),
        "pct_rater_annotations_multi_select": round(100 * pct_rows, 2),
        "pct_items_with_any_multi_select": round(100 * float(item_has_multi.mean()), 2),
    }


# ------------------------------------------------------------ deduplication

def normalize(t):
    t = str(t).lower()
    t = re.sub(r"\[name\]|\[religion\]", " ", t)
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


class _UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


COMMIT_RE = re.compile(r"[0-9a-fA-F]{40}")


def resolve_model_revision(model, model_name=None, revision=None):
    """Recover the commit the loaded model actually came from.

    Tried in order, because no single route works across all
    sentence-transformers / transformers versions, and a pinned revision that
    cannot be verified must not become an unopenable door:
      1. the config's cached commit hash
      2. the snapshot directory name in the resolved model path
      3. the hub API, as a last resort
    Returns None only if every route fails.
    """
    try:
        h = model[0].auto_model.config._commit_hash
        if h and COMMIT_RE.fullmatch(str(h)):
            return str(h)
    except Exception:
        pass
    try:
        path = str(model[0].auto_model.config.name_or_path)
        m = re.search(r"snapshots[/\\]([0-9a-fA-F]{40})", path)
        if m:
            return m.group(1)
    except Exception:
        pass
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(model_name or DEDUP_MODEL, revision=revision)
        if info.sha and COMMIT_RE.fullmatch(str(info.sha)):
            return str(info.sha)
    except Exception:
        pass
    return None


def require_pinned_dedup_model():
    """Freeze-quality runs must pin the embedding model to an immutable commit.

    The model is part of dataset construction: if it changes, the surviving
    corpus changes, the content fingerprint changes, and every hash in the
    freeze chain fails.

    A non-empty string is not sufficient. Branch names such as "main" or tags
    are mutable references - pinning to one looks pinned while still resolving
    to whatever the hub currently serves. Only a full 40-character commit SHA
    is accepted, and it is additionally verified against the revision the
    loaded model actually resolved to.
    """
    if DEDUP_MODEL_REVISION is None:
        raise SystemExit(
            "[FATAL] DEDUP_MODEL_REVISION is not pinned. Run once with "
            "--allow-unpinned-dedup (and WITHOUT --skip-near-dedup) to "
            "discover the resolved commit, recorded as "
            "model_revision_resolved, set the full 40-character SHA in "
            "corpus.py, then re-run from scratch.")
    if not COMMIT_RE.fullmatch(str(DEDUP_MODEL_REVISION)):
        raise SystemExit(
            f"[FATAL] DEDUP_MODEL_REVISION = {DEDUP_MODEL_REVISION!r} is not a "
            "full 40-character commit SHA. Branch names and tags are mutable "
            "and do not pin anything; a frozen build requires an immutable "
            "commit.")


def deduplicate(df, near_pass=True, threshold=DEDUP_THRESHOLD):
    """Exact/normalized dedup, then a connected-components near-duplicate pass.

    Applied ONCE to the base corpus, before any split, so R_clean and
    R_overlap differ in author overlap alone.
    """
    df = df.copy()
    df["_norm"] = df["text"].map(normalize)
    n0 = len(df)
    df = df[df["_norm"].str.len() > 0].drop_duplicates("_norm", keep="first")
    n_exact = n0 - len(df)
    n_near, ran, verified = 0, False, False

    if near_pass:
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.neighbors import NearestNeighbors
            print(f"[dedup] encoding with {DEDUP_MODEL} "
                  f"(revision={DEDUP_MODEL_REVISION or 'UNPINNED'}) ...")
            if DEDUP_MODEL_REVISION is None:
                print("[dedup] WARNING: embedding model revision is not "
                      "pinned. The corpus - and therefore every hash in the "
                      "freeze chain - depends on whichever version is "
                      "resolved. Pin DEDUP_MODEL_REVISION before a frozen run.")
            model = SentenceTransformer(DEDUP_MODEL,
                                        revision=DEDUP_MODEL_REVISION)
            resolved = resolve_model_revision(model, DEDUP_MODEL,
                                              DEDUP_MODEL_REVISION)
            if DEDUP_MODEL_REVISION is not None:
                # An unverifiable pin is not a pin. Passing the 40-character
                # format check while the loaded model's commit stays unknown
                # would let freeze_quality claim a verification that never
                # happened.
                if resolved is None:
                    raise SystemExit(
                        "[FATAL] The dedup model commit could not be verified "
                        "against the resolved model revision. Install "
                        "huggingface_hub, or upgrade sentence-transformers, so "
                        "the loaded commit can be read back.")
                if str(resolved).lower() != str(DEDUP_MODEL_REVISION).lower():
                    raise SystemExit(
                        f"[FATAL] Dedup model revision mismatch: requested "
                        f"{DEDUP_MODEL_REVISION}, resolved {resolved}. The "
                        "pinned commit is not the one actually loaded.")
                verified = True
            emb = model.encode(df["text"].tolist(), batch_size=256,
                               show_progress_bar=True,
                               normalize_embeddings=True)
            k = min(DEDUP_NEIGHBORS, len(df))
            nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(emb)
            dist, idx = nn.kneighbors(emb)
            uf = _UnionFind(len(df))
            for i in range(len(df)):
                for d, j in zip(dist[i][1:], idx[i][1:]):
                    if (1.0 - d) > threshold:
                        uf.union(i, int(j))
            keep = {}
            for i in range(len(df)):
                keep.setdefault(uf.find(i), i)
            keep_idx = sorted(keep.values())
            n_near = len(df) - len(keep_idx)
            df = df.iloc[keep_idx]
            ran = True
        except ImportError:
            print("[dedup] sentence-transformers/scikit-learn unavailable; "
                  "EXACT-ONLY dedup.")

    print(f"[dedup] exact {n_exact:,} + near {n_near:,} removed "
          f"-> {len(df):,} examples")
    return df.drop(columns=["_norm"]), {
        "exact_removed": int(n_exact), "near_removed": int(n_near),
        "near_pass_run": bool(ran),
        "algorithm": (f"exact match on normalised text, then connected "
                      f"components over a {DEDUP_NEIGHBORS}-nearest-neighbour "
                      f"cosine graph with edges above {threshold}; one item "
                      f"retained per component. Not an exhaustive all-pairs "
                      f"search."),
        "threshold": threshold,
        "neighbors": DEDUP_NEIGHBORS,
        "model": DEDUP_MODEL,
        "model_revision_requested": DEDUP_MODEL_REVISION,
        "model_revision_resolved": locals().get("resolved"),
        "model_revision_verified": bool(verified)}


def clean_authors(df):
    df = df.copy()
    df["author"] = df["author"].astype(str).str.strip()
    bad = df["author"].str.lower().isin(INVALID_AUTHORS)
    if bad.any():
        print(f"[auth ] dropped {int(bad.sum()):,} placeholder-author rows")
    return df[~bad]


def add_matching_keys(df):
    df = df.copy()
    # Normalise the timestamp to an integer before it reaches the fingerprint:
    # a float's text representation can differ across pandas versions, which
    # would change the corpus hash for no substantive reason.
    if "created_utc" in df.columns:
        df["created_utc_int"] = (pd.to_numeric(df["created_utc"], errors="coerce")
                                 .fillna(-1).astype("int64"))
    else:
        df["created_utc_int"] = -1
    tok = df["text"].astype(str).str.split().str.len()
    df["len_bucket"] = pd.qcut(tok, 4, labels=False, duplicates="drop")
    ppa = df.groupby("author")["id"].transform("size")
    df["author_n_posts"] = ppa
    df["ppa_bucket"] = pd.cut(ppa, [0, 1, 2, 4, 9, np.inf],
                              labels=["1", "2", "3-4", "5-9", "10+"]).astype(str)
    if "link_id" not in df.columns:
        raise ValueError("link_id is required for thread-disjoint eligibility")

    # A post without a usable thread id cannot be shown to be thread-disjoint
    # from a test target, so its author cannot be H1-eligible. It is NOT
    # dropped: it remains valid training and evaluation data, and discarding
    # it would waste corpus for no gain.
    lid = df["link_id"].astype(str).str.strip()
    df["link_id"] = lid
    df["link_id_valid"] = ~(lid.isin(["", "nan", "None", "<NA>"]) | lid.isna())
    n_bad = int((~df["link_id_valid"]).sum())
    if n_bad:
        print(f"[link ] {n_bad:,} posts have no usable link_id; their authors "
              f"are excluded from H1 eligibility but the posts are retained")
    return df


def build_corpus(data_dir, ekman_map_path, near_pass=True,
                 min_agreement=MIN_RATER_AGREEMENT):
    raw, raw_hashes = load_raw(data_dir)
    ekman_map, map_src = load_ekman_map(ekman_map_path)
    rater_ek, meta = rater_level_ekman(raw, ekman_map)
    single, multi, label_stats = aggregate_to_examples(rater_ek, meta,
                                                       min_agreement)
    # Author cleaning precedes deduplication. If a duplicate cluster contains
    # both a placeholder-author copy and a valid-author copy, cleaning last
    # can lose BOTH: dedup may keep the placeholder copy as the cluster
    # representative, and clean_authors then removes it. Cleaning first means
    # the representative is always drawn from valid authors.
    #
    # Side effect to record in the dataset card: a text appearing once under a
    # deleted author and once under a valid author is now always attributed to
    # the valid author, which shifts the author distribution slightly.
    single = clean_authors(single)
    corpus, dedup_stats = deduplicate(single, near_pass)
    corpus = add_matching_keys(corpus).reset_index(drop=True)
    corpus = corpus.set_index("id", drop=False)

    # The multi-label subset gets the same exact-dedup and author cleaning as
    # the primary corpus, so the two are comparable. It is a reference
    # artifact for reporting, not a second analysis corpus.
    multi_ref = (clean_authors(multi)
                 .assign(_norm=lambda x: x["text"].map(normalize))
                 .drop_duplicates("_norm", keep="first")
                 .drop(columns=["_norm"]))

    agreement = compute_agreement(rater_ek)
    info = {"raw_file_sha256": raw_hashes,
            "ekman_map_source": map_src,
            "ekman_map_sha256": sha_file(ekman_map_path),
            "label_construction": label_stats,
            "deduplication": dedup_stats, "agreement": agreement,
            "corpus_content_sha256": corpus_fingerprint(corpus),
            "corpus_ids_sha256": sha(corpus["id"].tolist()),
            "fingerprint_columns": FINGERPRINT_COLS}
    return corpus, multi_ref, info


# ------------------------------------------------------- split construction

def eligibility(df, authors=None):
    """Two eligibility definitions, both reported.

    strict  : >=2 posts spanning >=2 distinct link_ids. Required for H1,
              because a reserve post from the SAME thread as the test target
              injects conversational context as well as author identity -
              which would make H1 measure author + thread overlap rather than
              author overlap alone.
    loose   : >=2 posts, threads ignored. Reported for comparison only.
    """
    sub = df if authors is None else df[df["author"].isin(authors)]
    n_posts = sub.groupby("author").size()
    loose = set(n_posts[n_posts >= 2].index)
    # Thread counting uses only posts with a valid link_id; an author whose
    # posts carry no usable thread id cannot satisfy thread-disjointness.
    ok = sub[sub["link_id_valid"]] if "link_id_valid" in sub.columns else sub
    n_threads = ok.groupby("author")["link_id"].nunique()
    n_ok_posts = ok.groupby("author").size()
    cand = n_threads.reindex(n_posts.index).fillna(0)
    okp = n_ok_posts.reindex(n_posts.index).fillna(0)
    strict = set(n_posts[(okp >= 2) & (cand >= 2)].index)
    return strict, loose


def build_r_clean(df, seed):
    """Author-disjoint split with a thread-disjoint reserve pool.

    Every test author yields exactly one test post, so test size equals the
    number of test authors and the ratio targets are solved on the ANALYSIS
    corpus (train + val + test) after reserve and excluded items are removed.

    Roles: train / val / test / reserve / excluded.
    'excluded' holds a test author's remaining posts from the SAME thread as
    their test target - they cannot go to test (one target per author) and
    must not go to reserve (thread confound).
    """
    rng = np.random.default_rng(seed)
    sizes = df.groupby("author").size().sort_index()
    authors = sizes.index.to_numpy()
    counts = sizes.to_numpy()
    perm = rng.permutation(len(authors))
    authors, counts = authors[perm], counts[perm]
    total = int(counts.sum())

    n_test, drained, cut = 0, 0, 0
    for i, n in enumerate(counts):
        n_test += 1
        drained += int(n)
        analysis = (total - drained) + n_test
        cut = i + 1
        if analysis > 0 and n_test / analysis >= TARGET_TEST:
            break
    test_authors = set(authors[:cut])
    analysis = (total - drained) + n_test

    val_target = TARGET_VAL * analysis
    val_authors, acc = set(), 0
    for a, n in zip(authors[cut:], counts[cut:]):
        if acc >= val_target:
            break
        val_authors.add(a)
        acc += int(n)

    d = df.copy()
    d["partition"] = np.where(d["author"].isin(test_authors), "test",
                     np.where(d["author"].isin(val_authors), "val", "train"))
    d["role"] = d["partition"]

    strict, loose = eligibility(d[d["partition"] == "test"])
    eligible = strict

    reserve_idx, excluded_idx = [], []
    for author, grp in d[d["partition"] == "test"].groupby("author"):
        if author in eligible:
            valid = grp[grp["link_id_valid"]]
            tgt_thread = rng.choice(valid["link_id"].unique())
            in_thread = valid[valid["link_id"] == tgt_thread]
            target = in_thread.sample(n=1, random_state=seed).index[0]
            # same thread as the target, or no usable thread id -> excluded
            excluded_idx += [i for i in grp.index
                             if i != target and (
                                 not grp.loc[i, "link_id_valid"]
                                 or grp.loc[i, "link_id"] == tgt_thread)]
            reserve_idx += [i for i in valid.index
                            if valid.loc[i, "link_id"] != tgt_thread]
        else:
            target = grp.sample(n=1, random_state=seed).index[0]
            excluded_idx += [i for i in grp.index if i != target]

    d.loc[reserve_idx, "role"] = "reserve"
    d.loc[excluded_idx, "role"] = "excluded"

    frac = len(reserve_idx) / total if total else 0.0
    if frac > RESERVE_WARN_FRACTION:
        print(f"[seed{seed}] WARNING: reserve pool is {100*frac:.1f}% of the "
              f"corpus; heavy-tailed authors are draining the analysis set.")
    return d, eligible, {"strict_eligible": len(strict),
                         "loose_eligible": len(loose),
                         "excluded_same_thread": len(excluded_idx)}


def _match_index(train, rng):
    """Key hierarchy. Label is level 0 of every key and is NEVER relaxed,
    which holds per-class counts identical by construction."""
    levels = [
        lambda r: (r["label"], r["subreddit"], r["len_bucket"], r["ppa_bucket"]),
        lambda r: (r["label"], r["subreddit"], r["len_bucket"]),
        lambda r: (r["label"], r["subreddit"]),
        lambda r: (r["label"],),
    ]
    idxs = [defaultdict(list) for _ in levels]
    for idx, r in train.iterrows():
        for lv, keyfn in enumerate(levels):
            idxs[lv][keyfn(r)].append(idx)
    for d in idxs:
        for k in d:
            rng.shuffle(d[k])
    return levels, idxs


MATCH_TAGS = ["exact_4var", "no_ppa", "no_len", "label_only"]


def build_r_overlap(d, eligible, dose_k, seed):
    rng = np.random.default_rng(seed + 11)
    reserve = d[d["role"] == "reserve"]
    train = d[d["role"] == "train"]

    inject, skipped = [], 0
    for author, grp in reserve.groupby("author"):
        if len(grp) >= dose_k:
            inject += list(grp.sample(n=dose_k, random_state=seed).index)
        else:
            skipped += 1

    levels, idxs = _match_index(train, rng)
    used, removed = set(), []
    fallback = {t: 0 for t in MATCH_TAGS} | {"failed": 0}

    for idx in inject:
        r = d.loc[idx]
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

    overlap_train = sorted((set(train.index) - set(removed)) | set(inject))
    fallback["authors_short_of_dose"] = skipped
    return overlap_train, sorted(inject), sorted(removed), fallback


def build_r_random(df, seed):
    """Stratified random split.

    Called twice by the builder, on two deliberately different populations:

      r_random       - the ANALYSIS population (train + val + test), which is
                       size-matched to R_clean, so an R_random vs R_clean
                       contrast is not confounded by training-set size.
      r_random_full  - the whole cleaned corpus, for comparability with the
                       literature, which splits everything.

    Both are cheap index files; producing only one would either break the
    size match or break literature comparability.
    """
    rng = np.random.default_rng(seed + 90000)
    parts = {"train": [], "val": [], "test": []}
    for _, grp in df.groupby("label"):
        idx = grp.index.to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        a = int((1 - TARGET_VAL - TARGET_TEST) * n)
        b = int((1 - TARGET_TEST) * n)
        parts["train"] += list(idx[:a])
        parts["val"] += list(idx[a:b])
        parts["test"] += list(idx[b:])
    return {k: sorted(v) for k, v in parts.items()}


def complementary_blocks(d, seed):
    """Balanced author blocks A/B/C for jurors A+B, A+C, B+C, balanced on
    both total size and per-class composition."""
    rng = np.random.default_rng(seed + 7)
    train = d[d["role"] == "train"]
    per_author = (train.groupby(["author", "label"]).size()
                  .unstack(fill_value=0).reindex(columns=EKMAN, fill_value=0))
    order = per_author.sum(axis=1).sort_values(ascending=False).index

    blocks = {"A": [], "B": [], "C": []}
    load = {b: np.zeros(len(EKMAN)) for b in blocks}
    for author in order:
        vec = per_author.loc[author].to_numpy(dtype=float)
        best, best_cost = None, None
        for b in blocks:
            trial = {k: (load[k] + vec if k == b else load[k]) for k in blocks}
            cost = np.stack(list(trial.values())).std(axis=0).sum() \
                   + 0.001 * rng.random()
            if best_cost is None or cost < best_cost:
                best, best_cost = b, cost
        blocks[best].append(author)
        load[best] = load[best] + vec

    dist = {b: dict(zip(EKMAN, load[b].astype(int).tolist())) for b in blocks}
    totals = {b: int(load[b].sum()) for b in blocks}
    shares = np.stack([load[b] / max(load[b].sum(), 1) for b in blocks])
    drift = float((shares.max(axis=0) - shares.min(axis=0)).max())
    return blocks, {"examples_per_block": totals, "class_per_block": dist,
                    "max_class_share_drift": round(drift, 4)}


def verify_invariants(d, overlap_train, inject, removed, eligible, dose_k):
    clean_train = d[d["role"] == "train"]
    ov = d.loc[overlap_train]
    inj_authors = set(d.loc[inject, "author"]) if len(inject) else set()
    test_thread = (d[d["role"] == "test"].set_index("author")["link_id"].to_dict())
    thread_ok = all(d.loc[i, "link_id"] != test_thread.get(d.loc[i, "author"])
                    for i in inject)
    return {
        "train_size_identical": bool(len(clean_train) == len(ov)),
        "class_counts_identical": bool(
            clean_train["label"].value_counts().sort_index()
            .equals(ov["label"].value_counts().sort_index())),
        "injected_all_from_reserve": bool(
            (d.loc[inject, "role"] == "reserve").all()) if len(inject) else True,
        "removed_all_from_clean_train": bool(
            (d.loc[removed, "role"] == "train").all()) if len(removed) else True,
        "no_test_target_in_overlap_train": bool(
            not set(overlap_train) & set(d.index[d["role"] == "test"])),
        "uniform_author_coverage": bool(
            len(inject) == len(eligible) * dose_k
            and inj_authors == set(eligible)),
        "injections_thread_disjoint_from_target": bool(thread_ok),
    }


def write_table(df, path_no_ext):
    """Parquet when an engine is available, gzipped CSV otherwise."""
    try:
        df.to_parquet(f"{path_no_ext}.parquet", index=False)
        return "parquet"
    except Exception:
        df.to_csv(f"{path_no_ext}.csv.gz", index=False, compression="gzip")
        return "csv.gz"
