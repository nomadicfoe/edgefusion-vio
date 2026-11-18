"""Evaluate an estimated trajectory against ground truth.

Reports ATE (Sim3 aligned), RPE, a drift proxy, and the official KITTI
translation/rotation error. Saves a plot and one row per run in
results/results_table.csv (a rerun of the same run name replaces its row).

Usage:
    python -m src.evaluate --run results/kitti_00_pnp
"""
import argparse
import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from evo.core import metrics, sync
from evo.tools import file_interface

from src.kitti_metric import KITTI_LENGTHS, kitti_errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="results/<name> directory")
    ap.add_argument("--delta", type=int, default=100, help="RPE delta in frames")
    ap.add_argument(
        "--kitti_lengths",
        default=",".join(str(x) for x in KITTI_LENGTHS),
        help="sub-sequence lengths in metres for the KITTI metric",
    )
    args = ap.parse_args()
    run = Path(args.run)
    lengths = tuple(int(x) for x in args.kitti_lengths.split(","))

    gt = file_interface.read_tum_trajectory_file(str(run / "traj_gt.tum"))
    est = file_interface.read_tum_trajectory_file(str(run / "traj_est.tum"))
    gt, est = sync.associate_trajectories(gt, est)

    # official KITTI metric: raw poses, no alignment
    k_t, k_r, k_n, path_len = kitti_errors(np.array(gt.poses_se3), np.array(est.poses_se3), lengths)

    # ATE: align with Sim(3) because monocular scale may be off
    est_aligned = copy.deepcopy(est)
    est_aligned.align(gt, correct_scale=True)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((gt, est_aligned))
    ate_rmse = ape.get_statistic(metrics.StatisticsType.rmse)

    # RPE over `delta` frames, unaligned
    rpe = metrics.RPE(
        metrics.PoseRelation.translation_part,
        delta=args.delta,
        delta_unit=metrics.Unit.frames,
        all_pairs=False,
    )
    rpe.process_data((gt, est))
    rpe_rmse = rpe.get_statistic(metrics.StatisticsType.rmse)

    # drift proxy: mean RPE over delta frames / mean GT distance over delta frames
    gt_xyz = gt.positions_xyz
    seg = np.linalg.norm(gt_xyz[args.delta:] - gt_xyz[:-args.delta], axis=1).mean()
    drift_pct = 100.0 * float(np.mean(rpe.error)) / seg

    print(f"GT path length: {path_len:.0f} m")
    print(f"ATE RMSE (Sim3 aligned): {ate_rmse:.3f} m")
    print(f"RPE RMSE (delta={args.delta}): {rpe_rmse:.3f} m")
    print(f"RPE drift proxy: {drift_pct:.2f} %  (not the KITTI metric)")
    if k_n:
        print(f"KITTI translation error: {k_t:.2f} %")
        print(f"KITTI rotation error: {k_r:.3f} deg/100m   ({k_n} sub-sequences)")
    else:
        print("KITTI metric: n/a, sequence shorter than the shortest length")

    plt.figure(figsize=(6, 6))
    plt.plot(gt.positions_xyz[:, 0], gt.positions_xyz[:, 2], label="ground truth")
    plt.plot(est_aligned.positions_xyz[:, 0], est_aligned.positions_xyz[:, 2], label="estimate (aligned)")
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.axis("equal")
    plt.legend()
    plt.title(run.name)
    plt.savefig(run / "trajectory.png", dpi=150, bbox_inches="tight")
    plt.close()

    table = run.parent / "results_table.csv"
    row = {
        "run": run.name,
        "n_frames": len(gt.positions_xyz),
        "ate_rmse_m": round(ate_rmse, 4),
        "rpe_rmse_m": round(rpe_rmse, 4),
        "drift_pct": round(drift_pct, 3),
        "kitti_t_pct": round(k_t, 3),
        "kitti_r_deg100m": round(k_r, 4),
    }
    df_new = pd.DataFrame([row])
    if table.exists():
        old = pd.read_csv(table)
        old = old[old["run"] != run.name]
        df_new = pd.concat([old, df_new], ignore_index=True)
    df_new.to_csv(table, index=False)
    print(f"saved row to {table}")


if __name__ == "__main__":
    main()
