"""Run monocular VO on a KITTI sequence and save a TUM-format trajectory.

Usage:
    python -m src.run_vo --config configs/kitti_00_orb.yaml
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import yaml

from src.kitti import KittiSequence
from src.vo import MonoVO


def save_tum(path, poses, timestamps):
    """TUM format: timestamp tx ty tz qx qy qz qw."""
    from scipy.spatial.transform import Rotation

    with open(path, "w") as f:
        for ts, T in zip(timestamps, poses):
            q = Rotation.from_matrix(T[:3, :3]).as_quat()  # x y z w
            t = T[:3, 3]
            f.write(f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    seq = KittiSequence(cfg["data_root"], cfg["sequence"])
    n = len(seq) if cfg.get("max_frames") is None else min(cfg["max_frames"], len(seq))
    vo = MonoVO(
        seq.K,
        n_features=cfg["n_features"],
        ratio=cfg["ratio"],
        ransac_thresh=cfg["ransac_thresh"],
        min_inliers=cfg["min_inliers"],
    )

    out_dir = Path(cfg["out_dir"]) / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    T_w = np.eye(4)
    poses = [T_w.copy()]
    frame_log = []  # per-frame stats, used later for the reliability model
    img_prev = seq.image(0)
    kp_prev, des_prev = vo.detect(img_prev)
    t0 = time.time()

    for i in range(1, n):
        img = seq.image(i)
        kp, des = vo.detect(img)
        p1, p2 = vo.match(kp_prev, des_prev, kp, des)
        R, t, n_inl = vo.relative_pose(p1, p2)

        failed = R is None or n_inl < vo.min_inliers
        if not failed:
            scale = seq.gt_step(i) if cfg["use_gt_scale"] else 1.0
            T_rel = np.eye(4)  # frame i-1 -> frame i
            T_rel[:3, :3] = R
            T_rel[:3, 3] = t * scale
            T_w = T_w @ np.linalg.inv(T_rel)
        # on failure keep last pose (constant position), flag it

        poses.append(T_w.copy())
        frame_log.append(
            {
                "frame": i,
                "n_kp": len(kp),
                "n_matches": 0 if p1 is None else len(p1),
                "n_inliers": n_inl,
                "failed": int(failed),
            }
        )
        kp_prev, des_prev = kp, des
        if i % 200 == 0:
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
