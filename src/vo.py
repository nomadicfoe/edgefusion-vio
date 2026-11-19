"""Classical monocular VO pieces: ORB matching, two-view pose, two-view tracker."""
import cv2
import numpy as np


class MonoVO:
    def __init__(self, K, n_features=3000, ratio=0.75, ransac_thresh=1.0, min_inliers=30):
        self.K = K
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ratio = ratio
        self.ransac_thresh = ransac_thresh
        self.min_inliers = min_inliers

    def detect(self, img):
        kp, des = self.orb.detectAndCompute(img, None)
        return kp, des

    def match_idx(self, kp1, des1, kp2, des2):
        """Ratio-test matches. Returns (idx1, idx2) index arrays, or (None, None)."""
        if des1 is None or des2 is None or len(des1) < 8 or len(des2) < 8:
            return None, None
        pairs = self.matcher.knnMatch(des1, des2, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < self.ratio * n.distance]
        if len(good) < 8:
            return None, None
        return (
            np.array([m.queryIdx for m in good]),
            np.array([m.trainIdx for m in good]),
        )

    def match(self, kp1, des1, kp2, des2):
        i1, i2 = self.match_idx(kp1, des1, kp2, des2)
        if i1 is None:
            return None, None
        p1 = np.float32([kp1[i].pt for i in i1])
        p2 = np.float32([kp2[i].pt for i in i2])
        return p1, p2

    def relative_pose(self, p1, p2):
        """Return (R, t_unit, n_inliers). Maps points of frame 1 into frame 2."""
        if p1 is None:
            return None, None, 0
        E, mask = cv2.findEssentialMat(
            p1, p2, self.K, method=cv2.RANSAC, prob=0.999, threshold=self.ransac_thresh
        )
        if E is None or E.shape != (3, 3):
            return None, None, 0
        n_inl, R, t, _ = cv2.recoverPose(E, p1, p2, self.K, mask=mask)
        return R, t.ravel(), int(n_inl)


class TwoViewTracker:
    """Chain two-view poses frame to frame. Baseline only.

    Scale comes from the ground-truth step, so this is not true monocular VO.
    """

    def __init__(self, K, cfg):
        self.vo = MonoVO(
            K,
            n_features=cfg["n_features"],
            ratio=cfg["ratio"],
            ransac_thresh=cfg["ransac_thresh"],
            min_inliers=cfg["min_inliers"],
        )
        self.use_gt_scale = cfg.get("use_gt_scale", True)
        self.T_wc = np.eye(4)
        self.prev = None

    def process(self, img, gt_step=None, depth=None):
        kp, des = self.vo.detect(img)
        if self.prev is None:
            self.prev = (kp, des)
            info = dict(n_kp=len(kp), n_matches=0, n_inliers=0, n_map=0, failed=0)
            return self.T_wc.copy(), info
        p1, p2 = self.vo.match(*self.prev, kp, des)
        R, t, n_inl = self.vo.relative_pose(p1, p2)
        failed = R is None or n_inl < self.vo.min_inliers
        if not failed:
            scale = gt_step if (self.use_gt_scale and gt_step is not None) else 1.0
            T_rel = np.eye(4)  # frame i-1 -> frame i
            T_rel[:3, :3] = R
            T_rel[:3, 3] = t * scale
            self.T_wc = self.T_wc @ np.linalg.inv(T_rel)
        # on failure keep the last pose and flag it
        self.prev = (kp, des)
        info = dict(
            n_kp=len(kp),
            n_matches=0 if p1 is None else len(p1),
            n_inliers=n_inl,
            n_map=0,
            failed=int(failed),
        )
        return self.T_wc.copy(), info
