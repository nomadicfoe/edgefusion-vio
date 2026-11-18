"""Train and evaluate the reliability model with leave-one-sequence-out.

Each sequence is held out once: train on the others, score the held-out one.
This is the honest test, because frames inside one sequence look alike.

The label is rebuilt from t_pct and r_deg in the dataset, so thresholds can be
changed here without rerunning make_labels.

Compared against simple one-feature rules, so we can see if the model adds anything.

Usage:
    python -m src.train_reliability --data results/reliability_pnp.csv
    python -m src.train_reliability --data results/reliability_pnp.csv --thr_t 20 --thr_r 2
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

META = ["run", "seq", "method", "frame", "t_pct", "r_deg", "bad"]

# score = how likely a frame is bad, from one feature only (higher = worse)
HEURISTICS = {
    "inlier_ratio": lambda d: -d["inlier_ratio_mean"],
    "n_inliers": lambda d: -d["n_inliers_mean"],
    "step_jump": lambda d: d["step_jump"],
    "failed_flag": lambda d: d["failed_recent"],
}


def safe_auc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def recall_at_fpr(y, s, fpr_target=0.1):
    if not 0 < y.sum() < len(y):
        return np.nan
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(fpr_target, fpr, tpr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--thr_t", type=float, default=10.0)
    ap.add_argument("--thr_r", type=float, default=1.0)
    ap.add_argument("--out", default=None, help="save out-of-fold predictions here")
    ap.add_argument("--extra", default=None, help="extra features CSV from src.extra_features")
    ap.add_argument("--drop", default=None, help="comma list; drop features whose name contains any of these")
    args = ap.parse_args()

    df = pd.read_csv(args.data, dtype={"seq": str})
    df = df.dropna(subset=["t_pct", "r_deg"]).reset_index(drop=True)
    if args.extra:
        extra = pd.read_csv(args.extra)
        df = df.merge(extra, on=["run", "frame"], how="left")
        covered = df["epi_med"].notna().mean() if "epi_med" in df else 0.0
        print(f"merged extra features, {100 * covered:.0f}% of frames have the epipolar cue")
    df["bad"] = ((df["t_pct"] > args.thr_t) | (df["r_deg"] > args.thr_r)).astype(int)
    feats = [c for c in df.columns if c not in META]
    if args.drop:
        drop = [d for d in args.drop.split(",") if d]
        gone = [c for c in feats if any(d in c for d in drop)]
        feats = [c for c in feats if c not in gone]
        print(f"dropped {len(gone)} features matching {drop}")
    seqs = sorted(df["seq"].unique())
    print(f"{len(df)} frames, {len(feats)} features, bad if t_pct>{args.thr_t} or r_deg>{args.thr_r}")
    print(f"overall bad rate {100 * df['bad'].mean():.1f}%, sequences {seqs}\n")

    df["score"] = np.nan
    rows = []
    for s in seqs:
        tr, te = df[df["seq"] != s], df[df["seq"] == s]
        model = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=4, class_weight="balanced", random_state=0
        )
        model.fit(tr[feats], tr["bad"])
        score = model.predict_proba(te[feats])[:, 1]
        df.loc[te.index, "score"] = score
        y = te["bad"].to_numpy()
        row = {
            "held_out": s,
            "frames": len(te),
            "bad_%": round(100 * y.mean(), 1),
            "AUC_model": safe_auc(y, score),
            "recall@10%FPR": recall_at_fpr(y, score),
            "AP_model": average_precision_score(y, score) if y.sum() else np.nan,
        }
        for name, f in HEURISTICS.items():
            row[f"AUC_{name}"] = safe_auc(y, f(te).to_numpy())
        rows.append(row)

    table = pd.DataFrame(rows)
    print(table.round(3).to_string(index=False))

    y = df["bad"].to_numpy()
    print("\npooled out-of-fold:")
    print(f"  model AUC {safe_auc(y, df['score']):.3f}   AP {average_precision_score(y, df['score']):.3f}"
          f"   (AP of a random guess = bad rate = {y.mean():.3f})")
    for name, f in HEURISTICS.items():
        print(f"  {name:13s} AUC {safe_auc(y, f(df).to_numpy()):.3f}")

    if args.out:
        df[["run", "seq", "frame", "t_pct", "r_deg", "bad", "score"]].to_csv(args.out, index=False)
        print(f"\nsaved out-of-fold predictions to {args.out}")


if __name__ == "__main__":
    main()
