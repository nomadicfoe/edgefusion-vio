"""Official KITTI odometry metric: translation error (%) and rotation error (deg/100m),
averaged over sub-sequences of 100, 200, ..., 800 m, starting every 10 frames.
Poses are camera-to-world 4x4 matrices. No trajectory alignment is applied.
"""
import numpy as np

KITTI_LENGTHS = (100, 200, 300, 400, 500, 600, 700, 800)


def kitti_errors(gt, est, lengths=KITTI_LENGTHS, step=10):
    gt = np.asarray(gt)
    est = np.asarray(est)
    n = len(gt)
    dist = np.zeros(n)
    dist[1:] = np.cumsum(np.linalg.norm(np.diff(gt[:, :3, 3], axis=0), axis=1))

    t_errs, r_errs = [], []
    for first in range(0, n, step):
        for L in lengths:
            last = int(np.searchsorted(dist, dist[first] + L))
            if last >= n:
                continue
            d_gt = np.linalg.inv(gt[first]) @ gt[last]
            d_est = np.linalg.inv(est[first]) @ est[last]
            err = np.linalg.inv(d_gt) @ d_est
            t_errs.append(np.linalg.norm(err[:3, 3]) / L)
            cos = np.clip((np.trace(err[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
            r_errs.append(np.arccos(cos) / L)

    if not t_errs:
        return float("nan"), float("nan"), 0, float(dist[-1])
    t_pct = 100.0 * float(np.mean(t_errs))
    r_deg_100m = float(np.degrees(np.mean(r_errs)) * 100.0)
    return t_pct, r_deg_100m, len(t_errs), float(dist[-1])
