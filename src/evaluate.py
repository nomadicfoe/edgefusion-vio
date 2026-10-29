"""Evaluate an estimated trajectory against ground truth with evo (ATE and RPE)
and save a summary CSV row plus plots.

Usage:
    python -m src.evaluate --run results/kitti_00_orb
"""
import argparse
import copy
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from evo.core import metrics, sync
from evo.tools import file_interface


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="results/<name> directory")
    ap.add_argument("--delta", type=int, default=100, help="RPE delta in frames")
    args = ap.parse_args()
    run = Path(args.run)

    gt = file_interface.read_tum_trajectory_file(str(run / "traj_gt.tum"))
    est = file_interface.read_tum_trajectory_file(str(run / "traj_est.tum"))
    gt, est = sync.associate_trajectories(gt, est)

    # ATE: align with Sim(3) because monocular scale may be off
    est_aligned = copy.deepcopy(est)
    est_aligned.align(gt, correct_scale=True)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((gt, est_aligned))
    ate_rmse = ape.get_statistic(metrics.StatisticsType.rmse)

    # RPE: translation error over `delta` frames (KITTI-style drift proxy)
    rpe = metrics.RPE(
        metrics.PoseRelation.translation_part,
        delta=args.delta,
        delta_unit=metrics.Unit.frames,
        all_pairs=False,
    )
    rpe.process_data((gt, est))
    rpe_rmse = rpe.get_statistic(metrics.StatisticsType.rmse)

    # drift in percent: mean RPE over delta divided by mean GT path length of delta frames
    gt_xyz = gt.positions_xyz
    seg = np.linalg.norm(gt_xyz[args.delta:] - gt_xyz[:-args.delta], axis=1).mean()
    drift_pct = 100.0 * float(np.mean(rpe.error)) / seg

    print(f"ATE RMSE (Sim3 aligned): {ate_rmse:.3f} m")
    print(f"RPE RMSE (delta={args.delta}): {rpe_rmse:.3f} m")
    print(f"Translational drift: {drift_pct:.2f} %")

    # plots
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

    # append to one shared results table so every experiment is one row
    table = run.parent / "results_table.csv"
    new = not table.exists()
    with open(table, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["run", "ate_rmse_m", "rpe_rmse_m", "drift_pct"])
        w.writerow([run.name, f"{ate_rmse:.4f}", f"{rpe_rmse:.4f}", f"{drift_pct:.3f}"])
    print(f"appended row to {table}")


if __name__ == "__main__":
    main()
