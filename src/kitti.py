"""KITTI Odometry loader (grayscale, left camera image_0)."""
from pathlib import Path

import cv2
import numpy as np


class KittiSequence:
    def __init__(self, root, seq="00"):
        self.root = Path(root)
        self.seq = seq
        self.seq_dir = self.root / "sequences" / seq
        self.image_dir = self.seq_dir / "image_0"
        self.images = sorted(self.image_dir.glob("*.png"))
        if not self.images:
            raise FileNotFoundError(f"No images in {self.image_dir}")
        self.K = self._load_K()
        self.gt = self._load_poses()  # (N, 4, 4) or None

    def _load_K(self):
        with open(self.seq_dir / "calib.txt") as f:
            for line in f:
                if line.startswith("P0:"):
                    p = np.array(line.split()[1:], dtype=np.float64).reshape(3, 4)
                    return p[:, :3].copy()
        raise ValueError("P0 not found in calib.txt")

    def _load_poses(self):
        path = self.root / "poses" / f"{self.seq}.txt"
        if not path.exists():
            return None
        raw = np.loadtxt(path).reshape(-1, 3, 4)
        poses = np.tile(np.eye(4), (len(raw), 1, 1))
        poses[:, :3, :] = raw
        return poses

    def __len__(self):
        return len(self.images)

    def image(self, i):
        return cv2.imread(str(self.images[i]), cv2.IMREAD_GRAYSCALE)

    def gt_step(self, i):
        """Distance between GT positions of frame i-1 and i (used as scale)."""
        return float(np.linalg.norm(self.gt[i][:3, 3] - self.gt[i - 1][:3, 3]))
