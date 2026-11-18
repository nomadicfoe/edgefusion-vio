"""Monocular VO with a small landmark map: triangulate, then track with PnP + RANSAC.

Each frame:
  1. match ORB features to the previous frame
  2. matches that already have a 3D landmark give 3D-2D pairs -> solvePnPRansac
  3. landmarks of PnP inliers are carried to the current frame
  4. every tracked feature remembers where it was first seen (pixel + camera pose).
     Its landmark is re-triangulated from that first sighting and the current frame.
     With forward motion, two neighbouring frames give almost no parallax, so the
     baseline has to grow with the age of the track.

Scale: plain PnP tracking shrinks its own scale over time, so the step length of every
frame comes from a scale source (config `scale_source`). Right now that is the
ground-truth step, the same crutch the baseline uses, so the two methods are comparable
and the difference shows what PnP and the map change (rotation and direction).
Replace it with a depth-based estimate later.
"""
import cv2
import numpy as np

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
        self.scale_source = cfg.get("scale_source", "gt")  # "gt" or "none"

        self.pose_hist = []  # world -> camera pose of every frame so far
        self.prev = None  # (kp, des, landmarks, ref_pt, ref_id) of the previous frame
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
        Returns (X_world (N,3), good mask (N,)).
        """
        n = len(p1)
        if n == 0:
            return np.zeros((0, 3)), np.zeros(0, dtype=bool)
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
            good &= np.arccos(np.clip(cosang, -1.0, 1.0)) > self.min_parallax
        return X, good

    def _fix_scale(self, T_new, T_prev, gt_step):
        """Keep PnP rotation and direction, set the step length from the scale source.

        Plain PnP tracking shrinks its own scale (noisy landmarks bias each step short,
        the next map is built from that short step, and so on). A scale source stops that.
        scale_source "gt": ground-truth step (same crutch as the baseline).
        scale_source "none": keep the PnP step (scale drifts).
        Later this is where a depth-based scale estimate goes.
        """
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

    # ---------- main step ----------
    def _out(self, T_cw, n_kp, n_matches, n_inliers, lm, failed):
        info = dict(
            n_kp=n_kp,
            n_matches=n_matches,
            n_inliers=n_inliers,
            n_map=int(np.isfinite(lm[:, 0]).sum()),
            failed=int(failed),
        )
        return np.linalg.inv(T_cw), info

    def process(self, img, gt_step=None):
        kp, des = self.vo.detect(img)
        n_kp = len(kp)
        pts_cur = np.float32([k.pt for k in kp]).reshape(-1, 2)
        fid = len(self.pose_hist)  # id of this frame
        lm1 = np.full((n_kp, 3), np.nan)
        ref_pt1 = pts_cur.copy()  # a new feature is its own first sighting
        ref_id1 = np.full(n_kp, fid)

        if self.prev is None:  # first frame: pose is identity, no map yet
            self.pose_hist.append(np.eye(4))
            self.prev = (kp, des, lm1, ref_pt1, ref_id1)
            return self._out(self.pose_hist[-1], n_kp, 0, 0, lm1, 0)

        T_prev = self.pose_hist[-1]
        kp0, des0, lm0, ref_pt0, ref_id0 = self.prev
        i0, i1 = self.vo.match_idx(kp0, des0, kp, des)
        if i0 is None:  # nothing matched: hold pose, map is lost
            self.pose_hist.append(T_prev.copy())
            self.prev = (kp, des, lm1, ref_pt1, ref_id1)
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
            else:
                T_new = T_prev.copy()  # hold pose
                failed = 1

        step = np.linalg.norm(cam_center(T_new) - cam_center(T_prev))
        if step > 1e-6:
            self.last_step = step
        self.pose_hist.append(T_new)

        # carry first-sighting info of tracked features, then re-triangulate them
        ref_pt1[i1] = ref_pt0[i0]
        ref_id1[i1] = ref_id0[i0]
        T1s = np.stack([self.pose_hist[k] for k in ref_id1[i1]])
        X, good = self._triangulate(T1s, T_new, ref_pt1[i1], pts1)
        lm1[i1[good]] = X[good]

        self.prev = (kp, des, lm1, ref_pt1, ref_id1)
        return self._out(T_new, n_kp, len(i0), n_inl, lm1, failed)
