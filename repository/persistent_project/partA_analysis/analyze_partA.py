#!/usr/bin/env python3
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import binomtest, norm

EXPECTED_CONFIG_SHA = "3751576124daa427dec2890ed77abf87eb8c347b857649122f253eaaf07cfc8f"
DEFAULT_CHECKPOINT = "/kaggle/input/datasets/mohkh84/emotion-parta-predictions-checkpoint45"
DEFAULT_ROOT = "/kaggle/working/emotion_project"
LABELS = ["anger","disgust","fear","joy","sadness","surprise"]
P_COLS = [f"p_{x}" for x in LABELS]
L2I = {x:i for i,x in enumerate(LABELS)}

def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def stable_seed(base, name):
    h = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    return (int(base) + h) % (2**32 - 1)

def softmax_weights(vals, T=1.0):
    x = np.asarray(vals, float) / float(T)
    x -= x.max()
    w = np.exp(x)
    return w / w.sum()

def load_run(cp, run_id):
    csv = cp / f"{run_id}.csv"
    meta = cp / f"{run_id}.meta.json"
    if not csv.exists() or not meta.exists():
        raise FileNotFoundError(run_id)
    m = json.loads(meta.read_text(encoding="utf-8"))
    d = pd.read_csv(csv)
    exp = ["example_id","true_label","pred_label",*P_COLS]
    if list(d.columns) != exp:
        raise AssertionError(f"schema mismatch: {run_id}")
    if m["run_id"] != run_id or m["prediction_file_sha256"] != sha256(csv):
        raise AssertionError(f"binding mismatch: {run_id}")
    probs = d[P_COLS].to_numpy(float)
    if not np.isfinite(probs).all() or not np.allclose(probs.sum(1), 1.0, atol=1e-5, rtol=1e-5):
        raise AssertionError(f"bad probabilities: {run_id}")
    arg = np.asarray(LABELS, object)[probs.argmax(1)]
    if not np.array_equal(arg, d["pred_label"].astype(str).to_numpy()):
        raise AssertionError(f"pred_label != argmax(prob): {run_id}")
    return d, m

def aligned(runs):
    ids = runs[0][0]["example_id"].astype(str).to_numpy()
    y = runs[0][0]["true_label"].astype(str).to_numpy()
    for d,_ in runs[1:]:
        if not np.array_equal(ids, d["example_id"].astype(str).to_numpy()):
            raise AssertionError("ID alignment failure")
        if not np.array_equal(y, d["true_label"].astype(str).to_numpy()):
            raise AssertionError("true-label alignment failure")
    return ids, y

def metric_bundle(y, pred):
    y = np.asarray(y)
    pred = np.asarray(pred)
    pc = f1_score(y, pred, labels=LABELS, average=None, zero_division=0)
    sup = [(y == lab).sum() for lab in LABELS]
    return {
        "accuracy": float(accuracy_score(y,pred)),
        "macro_f1": float(f1_score(y,pred,labels=LABELS,average="macro",zero_division=0)),
        "weighted_f1": float(f1_score(y,pred,labels=LABELS,average="weighted",zero_division=0)),
        "per_class_f1": {lab: float(v) for lab,v in zip(LABELS,pc)},
        "per_class_support": {lab: int(v) for lab,v in zip(LABELS,sup)},
    }

def macro_f1_np(y_idx, p_idx):
    k = len(LABELS)
    cm = np.bincount(y_idx*k + p_idx, minlength=k*k).reshape(k,k)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    den = 2*tp + fp + fn
    f1 = np.divide(2*tp, den, out=np.zeros_like(tp), where=den>0)
    return float(f1.mean())

def paired_perm_accuracy(y, a, b, n_perm, seed):
    ca = (np.asarray(a)==np.asarray(y)).astype(np.int8)
    cb = (np.asarray(b)==np.asarray(y)).astype(np.int8)
    d = ca - cb
    obs = float(d.mean())
    nz = d[d != 0]
    if len(nz) == 0:
        return {"delta": obs, "p_two_sided": 1.0}
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_perm):
        s = rng.choice(np.array([-1,1], dtype=np.int8), size=len(nz))
        stat = float((nz*s).sum() / len(d))
        if abs(stat) >= abs(obs) - 1e-15:
            extreme += 1
    return {"delta": obs, "p_two_sided": float((extreme+1)/(n_perm+1))}

def paired_perm_macro(y, a, b, n_perm, seed):
    yi = np.fromiter((L2I[x] for x in y), dtype=np.int16, count=len(y))
    ai = np.fromiter((L2I[x] for x in a), dtype=np.int16, count=len(a))
    bi = np.fromiter((L2I[x] for x in b), dtype=np.int16, count=len(b))
    obs = macro_f1_np(yi,ai) - macro_f1_np(yi,bi)
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_perm):
        swap = rng.integers(0,2,size=len(yi),dtype=np.int8).astype(bool)
        pa = np.where(swap, bi, ai)
        pb = np.where(swap, ai, bi)
        stat = macro_f1_np(yi,pa) - macro_f1_np(yi,pb)
        if abs(stat) >= abs(obs) - 1e-15:
            extreme += 1
    return {"delta": float(obs), "p_two_sided": float((extreme+1)/(n_perm+1))}

def bootstrap_delta(y, a, b, metric, n_boot, seed):
    y = np.asarray(y); a=np.asarray(a); b=np.asarray(b)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = np.empty(n_boot, float)
    for i in range(n_boot):
        ix = rng.integers(0,n,size=n)
        yy,aa,bb = y[ix],a[ix],b[ix]
        if metric == "accuracy":
            vals[i] = accuracy_score(yy,aa)-accuracy_score(yy,bb)
        elif metric == "macro_f1":
            vals[i] = (
                f1_score(yy,aa,labels=LABELS,average="macro",zero_division=0)
                - f1_score(yy,bb,labels=LABELS,average="macro",zero_division=0)
            )
        else:
            raise ValueError(metric)
    lo,hi = np.percentile(vals,[2.5,97.5])
    return [float(lo),float(hi)]

def mcnemar_exact(y, a, b):
    y=np.asarray(y); a=np.asarray(a); b=np.asarray(b)
    ca=a==y; cb=b==y
    a_only = int(np.sum(ca & ~cb))
    b_only = int(np.sum(~ca & cb))
    n = a_only+b_only
    p = 1.0 if n==0 else float(binomtest(a_only,n,0.5,alternative="two-sided").pvalue)
    return {"a_correct_b_wrong":a_only,"a_wrong_b_correct":b_only,"p_exact_two_sided":p}

def cohen_h_acc(y,a,b):
    p1=float(np.mean(np.asarray(a)==np.asarray(y)))
    p2=float(np.mean(np.asarray(b)==np.asarray(y)))
    h=2*math.asin(math.sqrt(p1))-2*math.asin(math.sqrt(p2))
    return {"accuracy_a":p1,"accuracy_b":p2,"h_signed_a_minus_b":float(h)}

def tost_paired_accuracy(y,a,b,margin=0.01,alpha=0.05):
    d=(np.asarray(a)==np.asarray(y)).astype(float)-(np.asarray(b)==np.asarray(y)).astype(float)
    delta=float(d.mean())
    if len(d) < 2:
        raise ValueError("TOST needs n>=2")
    se=float(d.std(ddof=1)/math.sqrt(len(d)))
    if se == 0:
        inside = (-margin < delta < margin)
        pl = pu = 0.0 if inside else 1.0
    else:
        z_lower=(delta+margin)/se
        z_upper=(delta-margin)/se
        pl=float(1-norm.cdf(z_lower))
        pu=float(norm.cdf(z_upper))
    return {
        "delta_accuracy":delta,"margin":float(margin),"se_paired":se,
        "p_lower":pl,"p_upper":pu,"equivalent":bool(pl<alpha and pu<alpha),
        "method":"paired-normal-approximation TOST on per-example correctness differences"
    }

def holm_adjust(pmap):
    items=sorted(pmap.items(), key=lambda kv:kv[1])
    m=len(items); out={}; running=0.0
    for i,(name,p) in enumerate(items):
        adj=min(1.0,(m-i)*float(p))
        running=max(running,adj)
        out[name]=running
    return {k:float(out[k]) for k in pmap}

def ensemble_from_runs(runs, weights):
    aligned(runs)
    probs=np.stack([d[P_COLS].to_numpy(float) for d,_ in runs],axis=0)
    w=np.asarray(weights,float)
    w=w/w.sum()
    ens=np.tensordot(w,probs,axes=(0,0))
    pred=np.asarray(LABELS,object)[ens.argmax(1)]
    return ens,pred

def hard_vote_sensitivity(runs):
    _,y=aligned(runs)
    preds=np.stack([d["pred_label"].astype(str).to_numpy() for d,_ in runs],axis=0)
    probs=np.stack([d[P_COLS].to_numpy(float) for d,_ in runs],axis=0).mean(0)
    out=[]
    for j in range(preds.shape[1]):
        vals,cts=np.unique(preds[:,j],return_counts=True)
        mx=cts.max()
        if mx>=2:
            out.append(vals[cts.argmax()])
        else:
            out.append(LABELS[int(probs[j].argmax())])
    return np.asarray(out,object)

def bootstrap_h4_gain(y, member_preds, ens_pred, n_boot, seed):
    y=np.asarray(y); ens_pred=np.asarray(ens_pred)
    M=[np.asarray(x) for x in member_preds]
    rng=np.random.default_rng(seed); n=len(y)
    vals=np.empty(n_boot,float)
    for i in range(n_boot):
        ix=rng.integers(0,n,size=n)
        yy=y[ix]
        ma=[accuracy_score(yy,m[ix]) for m in M]
        ea=accuracy_score(yy,ens_pred[ix])
        vals[i]=ea-max(ma)
    lo,hi=np.percentile(vals,[2.5,97.5])
    return [float(lo),float(hi)]

def verify_inputs(root, cp, cfg, expected_cfg_sha):
    if sha256(cfg) != expected_cfg_sha:
        raise AssertionError("analysis config hash mismatch")
    c=json.loads(cfg.read_text(encoding="utf-8"))
    b=c["bindings"]
    if sha256(root/"training_config.json") != b["training_config_sha256"]:
        raise AssertionError("training config changed")
    if sha256(root/"run_experiments.py") != b["run_experiments_sha256"]:
        raise AssertionError("runner changed")
    if sha256(root/"splits/dataset_manifest.json") != b["dataset_manifest_sha256"]:
        raise AssertionError("dataset manifest changed")
    if sha256(cp/"checkpoint_manifest.json") != b["checkpoint_manifest_sha256"]:
        raise AssertionError("checkpoint manifest changed")
    if sha256(cp/"CHECKPOINT.sha256") != b["checkpoint_checksums_sha256"]:
        raise AssertionError("checkpoint checksum file changed")
    lines=[x.strip() for x in (cp/"CHECKPOINT.sha256").read_text().splitlines() if x.strip()]
    if len(lines)!=90:
        raise AssertionError("expected 90 checkpoint checksum entries")
    for line in lines:
        h,name=line.split(maxsplit=1)
        p=cp/name.strip()
        if not p.exists() or sha256(p)!=h:
            raise AssertionError(f"checkpoint artifact mismatch: {name}")
    return c

def read_eligible(root, split):
    p=root/"splits"/f"seed{split}"/"eligible_test.txt"
    return {x.strip() for x in p.read_text().splitlines() if x.strip()}

def analyze_h1(root,cp,cfg):
    out={}
    nperm=cfg["statistics"]["paired_permutation"]["n_permutations"]
    nboot=cfg["statistics"]["bootstrap"]["n_resamples"]
    base=cfg["statistics"]["implementation_rng_seed"]
    for s in cfg["H1"]["splits"]:
        elig=read_eligible(root,s)
        rows=[]
        for seed in cfg["H1"]["model_seeds"]:
            ra=cfg["H1"]["overlap_run_template"].format(split=s,seed=seed)
            rb=cfg["H1"]["reference_run_template"].format(split=s,seed=seed)
            A=load_run(cp,ra); B=load_run(cp,rb)
            ids,y=aligned([A,B])
            mask=np.fromiter((x in elig for x in ids),dtype=bool,count=len(ids))
            if int(mask.sum()) != len(elig):
                raise AssertionError(f"H1 eligible mask mismatch split{s}")
            yy=y[mask]
            pa=A[0]["pred_label"].astype(str).to_numpy()[mask]
            pb=B[0]["pred_label"].astype(str).to_numpy()[mask]
            name=f"H1_split{s}_seed{seed}"
            perm=paired_perm_accuracy(yy,pa,pb,nperm,stable_seed(base,name+"_perm"))
            ci=bootstrap_delta(yy,pa,pb,"accuracy",nboot,stable_seed(base,name+"_boot"))
            rows.append({
                "model_seed":seed,"n":int(len(yy)),
                "accuracy_r_overlap":float(accuracy_score(yy,pa)),
                "accuracy_r_clean":float(accuracy_score(yy,pb)),
                "delta_accuracy_overlap_minus_clean":perm["delta"],
                "paired_permutation_p_descriptive":perm["p_two_sided"],
                "bootstrap_95pct_CI_delta_accuracy":ci,
                "mcnemar_descriptive":mcnemar_exact(yy,pa,pb),
                "cohen_h_descriptive":cohen_h_acc(yy,pa,pb),
            })
        deltas=[r["delta_accuracy_overlap_minus_clean"] for r in rows]
        out[str(s)]={"per_model_seed":rows,
                     "descriptive_delta_mean_across_seeds":float(np.mean(deltas)),
                     "descriptive_delta_range_across_seeds":[float(np.min(deltas)),float(np.max(deltas))],
                     "note":"model seeds are not independent inferential units"}
    return out

def analyze_h2_h3(cp,cfg):
    h2={}; h3={}
    nperm=cfg["statistics"]["paired_permutation"]["n_permutations"]
    nboot=cfg["statistics"]["bootstrap"]["n_resamples"]
    base=cfg["statistics"]["implementation_rng_seed"]
    splits=[cfg["split_roles"]["confirmatory_split"],*cfg["split_roles"]["prespecified_replication_splits"]]
    for s in splits:
        jr_ids=[x.format(split=s) for x in cfg["H2"]["jury_members"]]
        jurors=[load_run(cp,x) for x in jr_ids]
        ids,y=aligned(jurors)
        vals=[float(m["validation_macro_f1"]) for _,m in jurors]
        T=float(cfg["H2"]["jury_aggregation"]["temperature_T"])
        w=softmax_weights(vals,T)
        _,weighted_pred=ensemble_from_runs(jurors,w)
        _,equal_pred=ensemble_from_runs(jurors,[1/3]*3)
        hard_pred=hard_vote_sensitivity(jurors)
        baseline_id=cfg["H2"]["baseline_run_template"].format(split=s)
        baseline=load_run(cp,baseline_id)
        aligned([jurors[0],baseline])
        base_pred=baseline[0]["pred_label"].astype(str).to_numpy()

        name=f"H2_split{s}"
        perm=paired_perm_macro(y,weighted_pred,base_pred,nperm,stable_seed(base,name+"_perm"))
        ci=bootstrap_delta(y,weighted_pred,base_pred,"macro_f1",nboot,stable_seed(base,name+"_boot"))
        h2[str(s)]={
            "n":int(len(y)),
            "jury_member_validation_macro_f1":{rid:float(v) for rid,v in zip(jr_ids,vals)},
            "jury_weights":{rid:float(v) for rid,v in zip(jr_ids,w)},
            "weighted_jury":metric_bundle(y,weighted_pred),
            "baseline_roberta":metric_bundle(y,base_pred),
            "delta_macro_f1_jury_minus_baseline":perm["delta"],
            "paired_permutation_p_primary":perm["p_two_sided"],
            "bootstrap_95pct_CI_delta_macro_f1":ci,
            "mcnemar_accuracy_secondary":mcnemar_exact(y,weighted_pred,base_pred),
            "cohen_h_accuracy_secondary":cohen_h_acc(y,weighted_pred,base_pred),
            "hard_vote_sensitivity":metric_bundle(y,hard_pred),
        }

        name=f"H3_split{s}"
        perm3=paired_perm_macro(y,weighted_pred,equal_pred,nperm,stable_seed(base,name+"_perm"))
        ci3=bootstrap_delta(y,weighted_pred,equal_pred,"macro_f1",nboot,stable_seed(base,name+"_boot"))
        h3[str(s)]={
            "n":int(len(y)),
            "weights_validation_softmax":{rid:float(v) for rid,v in zip(jr_ids,w)},
            "weighted_system":metric_bundle(y,weighted_pred),
            "equal_weight_system":metric_bundle(y,equal_pred),
            "delta_macro_f1_weighted_minus_equal":perm3["delta"],
            "paired_permutation_p_primary":perm3["p_two_sided"],
            "bootstrap_95pct_CI_delta_macro_f1":ci3,
            "mcnemar_accuracy_secondary":mcnemar_exact(y,weighted_pred,equal_pred),
            "cohen_h_accuracy_secondary":cohen_h_acc(y,weighted_pred,equal_pred),
            "tost_accuracy_margin_0.01":tost_paired_accuracy(y,weighted_pred,equal_pred,margin=0.01,alpha=0.05),
        }
    pmap={
        "H2_primary_split42":h2["42"]["paired_permutation_p_primary"],
        "H3_primary_split42":h3["42"]["paired_permutation_p_primary"],
    }
    return h2,h3,holm_adjust(pmap)

def analyze_h4(cp,cfg):
    out=[]; nboot=cfg["statistics"]["bootstrap"]["n_resamples"]; base=cfg["statistics"]["implementation_rng_seed"]
    for s in cfg["H4"]["splits"]:
        for cname,c in cfg["H4"]["configurations"].items():
            rids=[x.format(split=s) for x in c["members"]]
            runs=[load_run(cp,x) for x in rids]
            ids,y=aligned(runs)
            _,ens=ensemble_from_runs(runs,[1/3]*3)
            preds=[d["pred_label"].astype(str).to_numpy() for d,_ in runs]
            errors=[(p!=y).astype(float) for p in preds]
            cors=[]
            for i,j in [(0,1),(0,2),(1,2)]:
                if errors[i].std()==0 or errors[j].std()==0:
                    r=None
                else:
                    r=float(np.corrcoef(errors[i],errors[j])[0,1])
                cors.append(r)
            member_acc=[float(accuracy_score(y,p)) for p in preds]
            ens_acc=float(accuracy_score(y,ens))
            gain=ens_acc-max(member_acc)
            ci=bootstrap_h4_gain(y,preds,ens,nboot,stable_seed(base,f"H4_{cname}_{s}_boot"))
            out.append({
                "split":int(s),"configuration":cname,"n":int(len(y)),
                "member_run_ids":rids,"member_accuracies":member_acc,
                "best_member_accuracy":float(max(member_acc)),
                "ensemble_accuracy":ens_acc,"ensemble_gain_over_best_member":float(gain),
                "pairwise_error_correlations":{"m1_m2":cors[0],"m1_m3":cors[1],"m2_m3":cors[2]},
                "mean_pairwise_error_correlation":float(np.mean([x for x in cors if x is not None])) if any(x is not None for x in cors) else None,
                "bootstrap_95pct_CI_gain":ci,
                "training_overlap_O":float(c["training_overlap_O"]),
                "diversity_source":c["diversity_source"],
            })
    return out

def write_outputs(outdir, cfg_sha, h1,h2,h3,holm,h4):
    outdir.mkdir(parents=True,exist_ok=True)
    full={
        "analysis_config_sha256":cfg_sha,
        "H1":h1,"H2":h2,"H3":h3,
        "Holm_adjusted_confirmatory_primary_p":holm,
        "H4":h4,
        "notes":{
            "split42":"confirmatory for H2/H3",
            "splits123_2024":"prespecified replication; p-values not combined",
            "H1":"exploratory only",
            "H4":"exploratory descriptive; no regression",
        }
    }
    (outdir/"partA_results.json").write_text(json.dumps(full,indent=2,sort_keys=True,allow_nan=False)+"\n")
    h1rows=[]
    for s,v in h1.items():
        for r in v["per_model_seed"]:
            h1rows.append({
                "split":int(s),"model_seed":r["model_seed"],"n":r["n"],
                "accuracy_overlap":r["accuracy_r_overlap"],"accuracy_clean":r["accuracy_r_clean"],
                "delta_accuracy":r["delta_accuracy_overlap_minus_clean"],
                "perm_p_descriptive":r["paired_permutation_p_descriptive"],
                "ci_low":r["bootstrap_95pct_CI_delta_accuracy"][0],
                "ci_high":r["bootstrap_95pct_CI_delta_accuracy"][1],
            })
    pd.DataFrame(h1rows).to_csv(outdir/"H1_exploratory.csv",index=False)
    rows=[]
    for hyp,obj in [("H2",h2),("H3",h3)]:
        for s,r in obj.items():
            if hyp=="H2":
                rows.append({"hypothesis":hyp,"split":int(s),"delta_macro_f1":r["delta_macro_f1_jury_minus_baseline"],
                             "perm_p":r["paired_permutation_p_primary"],
                             "ci_low":r["bootstrap_95pct_CI_delta_macro_f1"][0],"ci_high":r["bootstrap_95pct_CI_delta_macro_f1"][1]})
            else:
                rows.append({"hypothesis":hyp,"split":int(s),"delta_macro_f1":r["delta_macro_f1_weighted_minus_equal"],
                             "perm_p":r["paired_permutation_p_primary"],
                             "ci_low":r["bootstrap_95pct_CI_delta_macro_f1"][0],"ci_high":r["bootstrap_95pct_CI_delta_macro_f1"][1]})
    pd.DataFrame(rows).to_csv(outdir/"H2_H3_primary_summary.csv",index=False)
    pd.DataFrame(h4).to_csv(outdir/"H4_descriptive.csv",index=False)
    ladder=[r for r in h4 if r["configuration"] in {"bert_initialization_jury","complementary_partition_jury","disjoint_partition_jury"}]
    pd.DataFrame(ladder).to_csv(outdir/"H4_bert_overlap_ladder.csv",index=False)
    files=sorted([p for p in outdir.iterdir() if p.is_file() and p.name!="ANALYSIS_OUTPUT.sha256"])
    lines=[f"{sha256(p)}  {p.name}" for p in files]
    (outdir/"ANALYSIS_OUTPUT.sha256").write_text("\n".join(lines)+"\n")
    return full

def self_test():
    y=np.array(["anger","anger","joy","joy","fear","surprise"],object)
    a=np.array(["anger","joy","joy","joy","fear","surprise"],object)
    b=np.array(["anger","anger","joy","surprise","joy","surprise"],object)
    assert metric_bundle(y,a)["accuracy"] == 5/6
    w=softmax_weights([0.6,0.7,0.8],1.0)
    assert np.isclose(w.sum(),1.0) and np.all(w>0)
    h=holm_adjust({"x":0.01,"y":0.04})
    assert np.isclose(h["x"],0.02) and np.isclose(h["y"],0.04)
    pp=paired_perm_accuracy(y,a,b,200,123)
    assert -1 <= pp["delta"] <= 1 and 0 <= pp["p_two_sided"] <= 1
    t=tost_paired_accuracy(y,a,a,margin=0.01,alpha=0.05)
    assert t["equivalent"] is True
    print("SELF_TEST_PASS")

def main():
    ap=argparse.ArgumentParser()
    g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--self-test",action="store_true")
    g.add_argument("--run",action="store_true")
    ap.add_argument("--project-root",default=DEFAULT_ROOT)
    ap.add_argument("--checkpoint",default=DEFAULT_CHECKPOINT)
    ap.add_argument("--config",default=None)
    ap.add_argument("--out-dir",default=None)
    ap.add_argument("--confirm",default="")
    args=ap.parse_args()

    if args.self_test:
        self_test(); return

    if args.confirm != "RUN_FROZEN_PARTA_ANALYSIS":
        raise SystemExit("Refusing real test analysis without --confirm RUN_FROZEN_PARTA_ANALYSIS")

    root=Path(args.project_root)
    cp=Path(args.checkpoint)
    cfgp=Path(args.config) if args.config else root/"partA_analysis"/"partA_analysis_config.json"
    outdir=Path(args.out_dir) if args.out_dir else root/"partA_analysis"/"results"
    cfg=verify_inputs(root,cp,cfgp,EXPECTED_CONFIG_SHA)

    h1=analyze_h1(root,cp,cfg)
    h2,h3,holm=analyze_h2_h3(cp,cfg)
    h4=analyze_h4(cp,cfg)
    write_outputs(outdir,EXPECTED_CONFIG_SHA,h1,h2,h3,holm,h4)

    print("PART_A_ANALYSIS_COMPLETE")
    print("CONFIG_SHA", EXPECTED_CONFIG_SHA)
    print("RESULTS", outdir/"partA_results.json")
    print("OUTPUT_HASHES", outdir/"ANALYSIS_OUTPUT.sha256")
    print("H2_SPLIT42_HOLM", holm["H2_primary_split42"])
    print("H3_SPLIT42_HOLM", holm["H3_primary_split42"])

if __name__=="__main__":
    main()
