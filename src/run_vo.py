"""Run monocular VO on a KITTI sequence and save a TUM-format trajectory.

Usage:
    python -m src.run_vo --config configs/kitti_00_pnp.yaml

Config key `method`: "twoview" (frame-to-frame baseline) or "pnp" (landmark map + PnP).
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import yaml

from src.kitti import KittiSequence
from src.vo import TwoViewTracker
from src.vo_pnp import PnPTracker


def save_tum(path, poses, timestamps):
    """TUM format: timestamp tx ty tz qx qy qz qw."""
    from scipy.spatial.transform import Rotation

    with open(path, "w") as f:
        for ts, T in zip(timestamps, poses):
            q = Rotation.from_matrix(T[:3, :3]).as_quat()  # x y z w
            t = T[:3, 3]
            f.write(f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")


def build_tracker(cfg, K):
    method = cfg.get("method", "twoview")
    if method == "twoview":
        return TwoViewTracker(K, cfg)
    if method == "pnp":
        return PnPTracker(K, cfg)
    raise ValueError(f"unknown method: {method}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    seq = KittiSequence(cfg["data_root"], cfg["sequence"])
    n = len(seq) if cfg.get("max_frames") is None else min(cfg["max_frames"], len(seq))
    tracker = build_tracker(cfg, seq.K)

    out_dir = Path(cfg["out_dir"]) / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    poses = []
    frame_log = []  # per-frame stats, used later for the reliability model
    t0 = time.time()
    for i in range(n):
        img = seq.image(i)
        gt_step = seq.gt_step(i) if (seq.gt is not None and i > 0) else None
        T_wc, info = tracker.process(img, gt_step)
        poses.append(T_wc)
        if i > 0:
            frame_log.append({"frame": i, **info})
        if i > 0 and i % 200 == 0:
            print(f"frame {i}/{n}  fps={i / (time.time() - t0):.1f}")

    elapsed = time.time() - t0
    timestamps = np.arange(len(poses)) * 0.1  # KITTI is 10 Hz
    save_tum(out_dir / "traj_est.tum", poses, timestamps)
    if seq.gt is not None:
        save_tum(out_dir / "traj_gt.tum", seq.gt[: len(poses)], timestamps)

    with open(out_dir / "frame_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(frame_log[0].keys()))
        w.writeheader()
        w.writerows(frame_log)

    n_fail = sum(r["failed"] for r in frame_log)
    print(f"done: {len(poses)} frames, {n_fail} tracking failures, {len(poses) / elapsed:.1f} fps")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
