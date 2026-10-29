"""Classical monocular frame-to-frame VO: ORB, essential matrix, recoverPose."""
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

    def match(self, kp1, des1, kp2, des2):
        if des1 is None or des2 is None or len(des1) < 8 or len(des2) < 8:
            return None, None
        pairs = self.matcher.knnMatch(des1, des2, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < self.ratio * n.distance]
        if len(good) < 8:
            return None, None
        p1 = np.float32([kp1[m.queryIdx].pt for m in good])
        p2 = np.float32([kp2[m.trainIdx].pt for m in good])
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
