"""Extra per-frame cues that need only the raw images and the estimated trajectory.
No rerun of the VO needed.

Image cues (frame i):
    sharp      variance of the Laplacian (low = blur)
    bright     mean intensity
    contrast   intensity std
    tex_cells  share of grid cells that have visible texture

Flow cues (dense Farneback flow, frame i-1 to i):
    flow_med, flow_p90   flow magnitude in pixels

Flow vs geometry (the main cue):
    The estimated pose between frames i-1 and i implies an epipolar geometry.
    Honest flow must satisfy it. epi_* is the Sampson error (pixels) of the flow
    vectors against that geometry. If the pose is wrong, the error goes up.
        epi_med, epi_p90, epi_frac1 (share of points above 1 px), epi_n

Rolling versions over the last `window` frames end in _r.

Usage:
    python -m src.extra_features \
        --runs results/kitti_00_pnp_full results/kitti_01_pnp \
        --data_root /content/kitti --out results/extra_pnp.csv
Run dir names must look like kitti_<seq>_<method>[_full].
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.kitti import KittiSequence
from src.make_labels import load_tum


def skew(t):
    return np.array([[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]])


def sampson(F, p1, p2):
    """Sampson distance in pixels of correspondences p1 -> p2 (N,2) for fundamental matrix F."""
    x1 = np.c_[p1, np.ones(len(p1))]
    x2 = np.c_[p2, np.ones(len(p2))]
    Fx2 = x2 @ F.T
    Ftx1 = x1 @ F
    num = np.sum(x1 * Fx2, axis=1) ** 2
    den = Fx2[:, 0] ** 2 + Fx2[:, 1] ** 2 + Ftx1[:, 0] ** 2 + Ftx1[:, 1] ** 2
    with np.errstate(all="ignore"):
        return np.sqrt(num / den)


def epipolar_stats(F, flow, g0s, scale, step, border, grad_thr):
    """Sample a grid on the half-size image, keep textured points, return Sampson errors."""
    h, w = g0s.shape
    ys, xs = np.mgrid[border:h - border:step, border:w - border:step]
    ys, xs = ys.ravel(), xs.ravel()
    gx = cv2.Sobel(g0s, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g0s, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2) / 8.0
    keep = mag[ys, xs] > grad_thr
    ys, xs = ys[keep], xs[keep]
    if len(xs) < 20:
        return None
    p1 = np.c_[xs, ys] / scale
    p2 = p1 + flow[ys, xs] / scale
    return sampson(F, p1, p2)


def frame_features(g0, g1, T_rel, K, scale=0.5, step=12, border=8, grad_thr=10.0):
    """Features for one frame pair. T_rel = inv(T_wc[i-1]) @ T_wc[i] (camera i in camera i-1)."""
    out = {}
    lap = cv2.Laplacian(g1, cv2.CV_64F)
    out["sharp"] = float(lap.var())
    out["bright"] = float(g1.mean())
    out["contrast"] = float(g1.std())
    h, w = g1.shape
    cells = np.array(
        [
            np.abs(cv2.Laplacian(g1[y:y + h // 4, x:x + w // 8], cv2.CV_64F)).mean()
            for y in range(0, h - h // 4 + 1, h // 4)
            for x in range(0, w - w // 8 + 1, w // 8)
        ]
    )
    out["tex_cells"] = float((cells > 4.0).mean())

    g0s = cv2.resize(g0, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    g1s = cv2.resize(g1, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    flow = cv2.calcOpticalFlowFarneback(g0s, g1s, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag = np.linalg.norm(flow, axis=2) / scale
    out["flow_med"] = float(np.median(mag))
    out["flow_p90"] = float(np.percentile(mag, 90))

    t = T_rel[:3, 3]
    nt = np.linalg.norm(t)
    err = None
    if nt > 1e-6:
        E = skew(t / nt) @ T_rel[:3, :3]
        Kinv = np.linalg.inv(K)
        F = Kinv.T @ E @ Kinv
        err = epipolar_stats(F, flow, g0s, scale, step, border, grad_thr)
    if err is None:
        out.update(epi_med=np.nan, epi_p90=np.nan, epi_frac1=np.nan, epi_n=0)
    else:
        out["epi_med"] = float(np.median(err))
        out["epi_p90"] = float(np.percentile(err, 90))
        out["epi_frac1"] = float((err > 1.0).mean())
        out["epi_n"] = int(len(err))
    return out


def run_features(run_dir, data_root, window):
    run = Path(run_dir)
    seq_id = run.name.split("_")[1]
    seq = KittiSequence(data_root, seq_id)
    est = load_tum(run / "traj_est.tum")
    n = min(len(est), len(seq))

    rows = [{"run": run.name, "frame": 0}]
    g0 = seq.image(0)
    for i in range(1, n):
        g1 = seq.image(i)
        T_rel = np.linalg.inv(est[i - 1]) @ est[i]
        f = frame_features(g0, g1, T_rel, seq.K)
        rows.append({"run": run.name, "frame": i, **f})
        g0 = g1
        if i % 500 == 0:
            print(f"  {run.name}: {i}/{n}")
    df = pd.DataFrame(rows)

    roll = lambda c: df[c].rolling(window, min_periods=1)
    df["epi_med_r"] = roll("epi_med").mean()
    df["epi_med_max_r"] = roll("epi_med").max()
    df["epi_p90_max_r"] = roll("epi_p90").max()
    df["epi_frac1_r"] = roll("epi_frac1").mean()
    df["sharp_min_r"] = roll("sharp").min()
    df["flow_med_r"] = roll("flow_med").mean()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=10)
    args = ap.parse_args()

    frames = []
    for r in args.runs:
        print(f"computing {r}")
        frames.append(run_features(r, args.data_root, args.window))
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(args.out, index=False)

    cols = ["sharp", "bright", "flow_med", "epi_med", "epi_p90", "epi_frac1", "epi_n"]
    print(data.groupby("run", sort=False)[cols].median().round(2).to_string())
    print(f"saved {len(data)} rows to {args.out}")


if __name__ == "__main__":
    main()
