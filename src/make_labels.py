"""Build the reliability dataset: one row per frame.

Label (needs ground truth, used only for training and scoring):
    local error over the last `window` frames, from the relative pose between
    frame i-window and frame i. No alignment needed.
        t_pct  = translation error / GT distance travelled * 100
        r_deg  = rotation error in degrees
        bad    = t_pct > thr_t or r_deg > thr_r

Features (no ground truth, available online):
    from frame_log.csv: n_kp, n_matches, n_inliers, n_map, failed, ratios,
    rolling mean/min/std over the window
    from the estimated trajectory: step length, rotation per frame, step jump

Usage:
    python -m src.make_labels \
        --runs results/kitti_00_pnp_full results/kitti_01_pnp results/kitti_05_pnp \
        --out results/reliability_pnp.csv
Run dir names must look like kitti_<seq>_<method>[_full].
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def load_tum(path):
    a = np.loadtxt(path)
    T = np.tile(np.eye(4), (len(a), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(a[:, 4:8]).as_matrix()  # x y z w
    T[:, :3, 3] = a[:, 1:4]
    return T


def local_errors(gt, est, window, d_floor=2.0):
    """Relative pose error over `window` frames, for every frame i >= window."""
    n = len(gt)
    t_pct = np.full(n, np.nan)
    r_deg = np.full(n, np.nan)
    for i in range(window, n):
        d_gt = np.linalg.inv(gt[i - window]) @ gt[i]
        d_est = np.linalg.inv(est[i - window]) @ est[i]
        err = np.linalg.inv(d_gt) @ d_est
        dist = np.linalg.norm(d_gt[:3, 3])
        t_pct[i] = 100.0 * np.linalg.norm(err[:3, 3]) / max(dist, d_floor)
        cos = np.clip((np.trace(err[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        r_deg[i] = np.degrees(np.arccos(cos))
    return t_pct, r_deg


def pose_features(est):
    """Step length and rotation per frame from the estimated trajectory."""
    n = len(est)
    step = np.zeros(n)
    rot = np.zeros(n)
    for i in range(1, n):
        d = np.linalg.inv(est[i - 1]) @ est[i]
        step[i] = np.linalg.norm(d[:3, 3])
        cos = np.clip((np.trace(d[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        rot[i] = np.degrees(np.arccos(cos))
    return step, rot


def build_run(run_dir, window, thr_t, thr_r):
    run = Path(run_dir)
    parts = run.name.split("_")
    seq, method = parts[1], parts[2]

    gt = load_tum(run / "traj_gt.tum")
    est = load_tum(run / "traj_est.tum")
    n = min(len(gt), len(est))
    gt, est = gt[:n], est[:n]

    t_pct, r_deg = local_errors(gt, est, window)
    step, rot = pose_features(est)

    log = pd.read_csv(run / "frame_log.csv").set_index("frame")
    df = pd.DataFrame({"frame": np.arange(n)})
    df = df.join(log, on="frame")
    df[["n_kp", "n_matches", "n_inliers", "n_map", "failed"]] = (
        df[["n_kp", "n_matches", "n_inliers", "n_map", "failed"]].fillna(0)
    )
    df["inlier_ratio"] = df["n_inliers"] / df["n_matches"].clip(lower=1)
    df["match_ratio"] = df["n_matches"] / df["n_kp"].clip(lower=1)
    df["est_step"] = step
    df["est_rot_deg"] = rot

    roll = lambda s: s.rolling(window, min_periods=1)
    for col in ("n_inliers", "inlier_ratio", "n_matches", "n_map"):
        df[f"{col}_mean"] = roll(df[col]).mean()
        df[f"{col}_min"] = roll(df[col]).min()
    df["n_inliers_std"] = roll(df["n_inliers"]).std().fillna(0)
    df["n_inliers_delta"] = df["n_inliers"].diff().fillna(0)
    med = roll(df["est_step"]).median().clip(lower=1e-3)
    df["step_jump"] = (df["est_step"] - med).abs() / med
    df["failed_recent"] = roll(df["failed"]).sum()

    df["t_pct"] = t_pct
    df["r_deg"] = r_deg
    df["bad"] = ((t_pct > thr_t) | (r_deg > thr_r)).astype(float)
    df.loc[np.isnan(t_pct), "bad"] = np.nan  # first `window` frames have no label
    df.insert(0, "method", method)
    df.insert(0, "seq", seq)
    df.insert(0, "run", run.name)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=10, help="frames (1 s on KITTI)")
    ap.add_argument("--thr_t", type=float, default=10.0, help="bad if local translation error > this %%")
    ap.add_argument("--thr_r", type=float, default=1.0, help="bad if local rotation error > this deg")
    args = ap.parse_args()

    frames = [build_run(r, args.window, args.thr_t, args.thr_r) for r in args.runs]
    data = pd.concat(frames, ignore_index=True)

    print(f"window={args.window} frames, bad if t_pct>{args.thr_t} or r_deg>{args.thr_r}")
    rows = []
    for name, g in data.groupby("run", sort=False):
        g = g.dropna(subset=["bad"])
        rows.append(
            {
                "run": name,
                "frames": len(g),
                "bad_%": round(100 * g["bad"].mean(), 1),
                "t_pct_med": round(g["t_pct"].median(), 1),
                "t_pct_p90": round(g["t_pct"].quantile(0.9), 1),
                "r_deg_med": round(g["r_deg"].median(), 2),
                "r_deg_p90": round(g["r_deg"].quantile(0.9), 2),
                "failed_flags": int(g["failed"].sum()),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))

    data.to_csv(args.out, index=False)
    print(f"saved {len(data)} rows to {args.out}")


if __name__ == "__main__":
    main()
