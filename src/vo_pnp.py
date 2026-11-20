"""Monocular VO with a small landmark map: triangulate, then track with PnP + RANSAC.

Each frame:
  1. match ORB features to the previous frame
  2. matches that already have a 3D landmark give 3D-2D pairs -> solvePnPRansac
  3. landmarks of PnP inliers are carried to the current frame
  4. every tracked feature remembers where it was first seen (pixel + camera pose).
     Its landmark is re-triangulated from that first sighting and the current frame.
     With forward motion, two neighbouring frames give almost no parallax, so the
     baseline has to grow with the age of the track.

Scale: plain PnP tracking shrinks its own scale over time, so scale needs an outside
source (config `scale_source`):
  "gt"     step length from the ground-truth step (baseline crutch, not real monocular)
  "depth"  metric depth from a monocular depth network. Each frame the median ratio
           between network depth and landmark depth gives a scale correction. The whole
           map is rescaled about the previous camera centre, so the recorded step
           length follows the network and the map stays consistent.
  "none"   keep the PnP step (scale drifts)
"""
import cv2
import numpy as np
from scipy.ndimage import map_coordinates

from src.vo import MonoVO


def cam_center(T_cw):
    return -T_cw[:3, :3].T @ T_cw[:3, 3]


class PnPTracker:
    def __init__(self, K, cfg):
        self.K = K.astype(np.float64)
        self.Kinv = np.linalg.inv(self.K)
        self.vo = MonoVO(
            K,
            n_features=cfg["n_features"],
            ratio=cfg["ratio"],
            ransac_thresh=cfg["ransac_thresh"],
            min_inliers=cfg["min_inliers"],
        )
        self.min_inliers = cfg["min_inliers"]
        self.pnp_thresh = cfg.get("pnp_thresh", 2.0)  # px
        self.min_parallax = np.radians(cfg.get("min_parallax_deg", 0.5))
        self.max_depth = cfg.get("max_depth", 150.0)  # m, ignore far points
        self.max_step = cfg.get("max_step", 4.0)  # m per frame, pose jump rule
        self.reproj_thresh = cfg.get("reproj_thresh", 2.0)  # px, triangulation check
        self.scale_source = cfg.get("scale_source", "gt")  # "gt", "depth" or "none"

        # depth scale settings
        self.depth_gain = cfg.get("depth_gain", 0.5)  # 1 = full correction each frame (unstable), 0.3-0.5 works
        self.depth_bias = cfg.get("depth_bias", 1.0)  # network depth is divided by this (network overestimates KITTI depth)
        self.depth_zmin = cfg.get("depth_zmin", 4.0)  # m, landmarks used for the ratio
        self.depth_zmax = cfg.get("depth_zmax", 50.0)
        self.depth_min_pts = cfg.get("depth_min_pts", 30)
        self.depth_min_parallax = np.radians(cfg.get("depth_min_parallax_deg", 2.0))  # only well-triangulated landmarks
        self.hist_keep = 60  # past poses rescaled with the map
        self.scale_ready = False

        self.pose_hist = []  # world -> camera pose of every frame so far
        self.prev = None  # (keypoints, descriptors, landmarks in world coords, ref_pt, ref_id)
        self.tracking = False  # becomes True after the first PnP success
        self.last_step = 1.0

    # ---------- geometry helpers ----------
    def _pnp(self, obj, img_pts):
        obj = np.ascontiguousarray(obj, dtype=np.float64)
        img_pts = np.ascontiguousarray(img_pts, dtype=np.float64)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            obj,
            img_pts,
            self.K,
            None,
            iterationsCount=200,
            reprojectionError=self.pnp_thresh,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok or inl is None or len(inl) < 6:
            return None, None
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img_pts[inl], self.K, None, rvec, tvec)
        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(rvec)[0]
        T[:3, 3] = tvec.ravel()
        return T, inl

    def _triangulate(self, T1s, T2, p1, p2):
        """DLT triangulation of N pixel pairs.

        T1s: (N,4,4) world->camera poses of the first views, T2: (4,4) of the second view.
        Returns (X_world (N,3), good mask (N,), parallax angle in radians (N,)).
        """
        n = len(p1)
        if n == 0:
            return np.zeros((0, 3)), np.zeros(0, dtype=bool), np.zeros(0)
        n1 = (np.c_[p1, np.ones(n)] @ self.Kinv.T)[:, :2]
        n2 = (np.c_[p2, np.ones(n)] @ self.Kinv.T)[:, :2]
        P1 = T1s[:, :3, :]
        P2 = np.broadcast_to(T2[:3], (n, 3, 4))
        A = np.stack(
            [
                n1[:, 0:1] * P1[:, 2] - P1[:, 0],
                n1[:, 1:2] * P1[:, 2] - P1[:, 1],
                n2[:, 0:1] * P2[:, 2] - P2[:, 0],
                n2[:, 1:2] * P2[:, 2] - P2[:, 1],
            ],
            axis=1,
        )
        Xh = np.linalg.svd(A)[2][:, -1, :]
        with np.errstate(all="ignore"):
            X = Xh[:, :3] / Xh[:, 3:4]
            c1 = np.einsum("nij,nj->ni", T1s[:, :3, :3], X) + T1s[:, :3, 3]
            c2 = X @ T2[:3, :3].T + T2[:3, 3]
            good = np.isfinite(X).all(axis=1)
            good &= (c1[:, 2] > 0.5) & (c2[:, 2] > 0.5)
            good &= (c1[:, 2] < self.max_depth) & (c2[:, 2] < self.max_depth)
            for c, p in ((c1, p1), (c2, p2)):
                uv = c @ self.K.T
                uv = uv[:, :2] / uv[:, 2:3]
                good &= np.linalg.norm(uv - p, axis=1) < self.reproj_thresh
            o1 = -np.einsum("nji,nj->ni", T1s[:, :3, :3], T1s[:, :3, 3])
            o2 = cam_center(T2)
            r1, r2 = X - o1, X - o2
            cosang = (r1 * r2).sum(1) / (np.linalg.norm(r1, axis=1) * np.linalg.norm(r2, axis=1))
            ang = np.arccos(np.clip(cosang, -1.0, 1.0))
            good &= ang > self.min_parallax
        return X, good, ang

    def _fix_scale(self, T_new, T_prev, gt_step):
        """scale_source "gt": keep PnP rotation and direction, set the step length to the GT step."""
        if self.scale_source != "gt" or gt_step is None:
            return T_new
        c_prev, c_new = cam_center(T_prev), cam_center(T_new)
        d = c_new - c_prev
        n = np.linalg.norm(d)
        if n < 1e-9:
            return T_new
        c_fixed = c_prev + d * (gt_step / n)
        T = T_new.copy()
        T[:3, 3] = -T[:3, :3] @ c_fixed
        return T

    # ---------- depth scale ----------
    def _depth_stats(self, lm, par, pts, T_cw, depth, img_shape):
        """Median ratio network depth / landmark depth in the current frame.

        Returns (median ratio, MAD of the log ratios, count) or None if too few points.
        """
        ok = np.isfinite(lm[:, 0]) & (par > self.depth_min_parallax)
        if ok.sum() < self.depth_min_pts:
            return None
        X, px = lm[ok], pts[ok]
        z = X @ T_cw[2, :3] + T_cw[2, 3]
        h, w = img_shape[:2]
        hd, wd = depth.shape
        yy = (px[:, 1] + 0.5) * hd / h - 0.5
        xx = (px[:, 0] + 0.5) * wd / w - 0.5
        zn = map_coordinates(depth.astype(np.float32), [yy, xx], order=1, mode="nearest")
        # select by the network depth, not by the noisy triangulated depth (that would bias the ratio)
        zn = zn / self.depth_bias
        good = np.isfinite(zn) & (zn > self.depth_zmin) & (zn < self.depth_zmax) & (z > 0.5)
        if good.sum() < self.depth_min_pts:
            return None
        lr = np.log(zn[good] / z[good])
        med = float(np.median(lr))
        mad = float(np.median(np.abs(lr - med)))
        return float(np.exp(med)), mad, int(good.sum())

    def _rescale_world(self, c0, rho, lm):
        """Scale the map and recent poses by rho about the camera centre c0. Returns the new current pose."""

        def scaled(T):
            c = c0 + rho * (cam_center(T) - c0)
            T2 = T.copy()
            T2[:3, 3] = -T[:3, :3] @ c
            return T2

        ok = np.isfinite(lm[:, 0])
        lm[ok] = c0 + rho * (lm[ok] - c0)
        for k in range(max(0, len(self.pose_hist) - self.hist_keep), len(self.pose_hist)):
            self.pose_hist[k] = scaled(self.pose_hist[k])
        return self.pose_hist[-1]

    # ---------- main step ----------
    def _out(self, T_cw, n_kp, n_matches, n_inliers, lm, failed, depth=(np.nan, np.nan, 0)):
        info = dict(
            n_kp=n_kp,
            n_matches=n_matches,
            n_inliers=n_inliers,
            n_map=int(np.isfinite(lm[:, 0]).sum()),
            failed=int(failed),
            depth_rho=depth[0],
            depth_mad=depth[1],
            depth_n=depth[2],
        )
        return np.linalg.inv(T_cw), info

    def process(self, img, gt_step=None, depth=None):
        kp, des = self.vo.detect(img)
        n_kp = len(kp)
        pts_cur = np.float32([k.pt for k in kp]).reshape(-1, 2)
        fid = len(self.pose_hist)  # id of this frame
        lm1 = np.full((n_kp, 3), np.nan)
        par1 = np.full(n_kp, np.nan)  # parallax (rad) of each landmark's triangulation
        ref_pt1 = pts_cur.copy()  # a new feature is its own first sighting
        ref_id1 = np.full(n_kp, fid)

        if self.prev is None:  # first frame: pose is identity, no map yet
            self.pose_hist.append(np.eye(4))
            self.prev = (kp, des, lm1, ref_pt1, ref_id1, par1)
            return self._out(self.pose_hist[-1], n_kp, 0, 0, lm1, 0)

        T_prev = self.pose_hist[-1]
        kp0, des0, lm0, ref_pt0, ref_id0, par0 = self.prev
        i0, i1 = self.vo.match_idx(kp0, des0, kp, des)
        if i0 is None:  # nothing matched: hold pose, map is lost
            self.pose_hist.append(T_prev.copy())
            self.prev = (kp, des, lm1, ref_pt1, ref_id1, par1)
            return self._out(T_prev, n_kp, 0, 0, lm1, 1)

        pts0 = np.float32([kp0[i].pt for i in i0])
        pts1 = pts_cur[i1]
        has_lm = np.isfinite(lm0[i0, 0])

        T_new, inl = None, None
        if has_lm.sum() >= 6:
            T_new, inl = self._pnp(lm0[i0[has_lm]], pts1[has_lm])
        if T_new is not None:
            T_new = self._fix_scale(T_new, T_prev, gt_step)
            step = np.linalg.norm(cam_center(T_new) - cam_center(T_prev))
            if len(inl) < self.min_inliers or step > self.max_step:
                T_new = None

        n_inl = 0
        failed = 0
        if T_new is not None:
            sel = np.where(has_lm)[0][inl]
            lm1[i1[sel]] = lm0[i0[sel]]  # carry landmarks of PnP inliers
            par1[i1[sel]] = par0[i0[sel]]
            n_inl = len(inl)
            self.tracking = True
        else:
            # map still being built, or tracking failure: two-view fallback
            failed = 1 if self.tracking else 0
            R, t, n2 = self.vo.relative_pose(pts0, pts1)
            if R is not None and n2 >= self.min_inliers:
                scale = gt_step if gt_step is not None else self.last_step
                T_rel = np.eye(4)
                T_rel[:3, :3] = R
                T_rel[:3, 3] = t * scale
                T_new = T_rel @ T_prev
                n_inl = n2
                lm1[i1[has_lm]] = lm0[i0[has_lm]]  # keep the map
                par1[i1[has_lm]] = par0[i0[has_lm]]
            else:
                T_new = T_prev.copy()  # hold pose
                failed = 1

        self.pose_hist.append(T_new)

        # carry first-sighting info of tracked features, then re-triangulate them
        ref_pt1[i1] = ref_pt0[i0]
        ref_id1[i1] = ref_id0[i0]
        T1s = np.stack([self.pose_hist[k] for k in ref_id1[i1]])
        X, good, ang = self._triangulate(T1s, T_new, ref_pt1[i1], pts1)
        lm1[i1[good]] = X[good]
        par1[i1[good]] = ang[good]

        # depth scale: compare network depth with landmark depth, rescale the map
        dstats = (np.nan, np.nan, 0)
        if self.scale_source == "depth" and depth is not None:
            stats = self._depth_stats(lm1, par1, pts_cur, T_new, depth, img.shape)
            if stats is not None:
                dstats = stats
                gain = self.depth_gain if self.scale_ready else 1.0
                T_new = self._rescale_world(cam_center(T_prev), stats[0] ** gain, lm1)
                self.scale_ready = True

        step = np.linalg.norm(cam_center(T_new) - cam_center(T_prev))
        if step > 1e-6:
            self.last_step = step
        self.prev = (kp, des, lm1, ref_pt1, ref_id1, par1)
        return self._out(T_new, n_kp, len(i0), n_inl, lm1, failed, dstats)
