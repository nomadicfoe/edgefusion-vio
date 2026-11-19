"""Cache monocular metric depth for KITTI sequences.

Runs a depth network once per frame and stores the maps at quarter resolution
(float16, one .npy file per sequence, shape N x H/4 x W/4). The tracker reads these
files, so the network never runs inside the VO loop.

Usage (Colab, GPU runtime):
    python -m src.depth_cache --data_root /content/kitti --seqs 00 01 02 \
        --out /content/drive/MyDrive/edgefusion-vio/depth

Finished sequences are skipped, so a disconnect only costs the sequence in progress.
The default model gives metric depth in metres for outdoor scenes.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from src.kitti import KittiSequence

DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model(name, device, in_h=252, in_w=826):
    """Return predict(list of HxW uint8 gray images) -> (B,H,W) float32 depth in metres.

    Images are resized to in_h x in_w (multiples of 14) and normalised here, so the
    result does not depend on processor settings. The output is resized back to the
    input size.
    """
    import torch
    from transformers import AutoModelForDepthEstimation

    model = AutoModelForDepthEstimation.from_pretrained(name).to(device).eval()
    half = device == "cuda"
    if half:
        model = model.half()

    @torch.no_grad()
    def predict(images):
        h, w = images[0].shape[:2]
        batch = []
        for im in images:
            rgb = np.repeat(im[:, :, None], 3, axis=2).astype(np.float32) / 255.0
            rgb = cv2.resize(rgb, (in_w, in_h), interpolation=cv2.INTER_CUBIC)
            batch.append(((rgb - MEAN) / STD).transpose(2, 0, 1))
        x = torch.from_numpy(np.stack(batch)).to(device)
        if half:
            x = x.half()
        out = model(pixel_values=x).predicted_depth  # (B, h', w') metres
        out = torch.nn.functional.interpolate(out[:, None].float(), size=(h, w), mode="bilinear", align_corners=False)
        return out[:, 0].cpu().numpy()

    return predict


def cache_sequence(seq, predict, out_path, scale=0.25, batch=8, max_frames=None):
    n = len(seq) if max_frames is None else min(max_frames, len(seq))
    h, w = seq.image(0).shape[:2]
    hd, wd = int(round(h * scale)), int(round(w * scale))
    out_path = Path(out_path)
    part = out_path.with_name(out_path.name + ".part")
    arr = np.lib.format.open_memmap(part, mode="w+", dtype=np.float16, shape=(n, hd, wd))
    t0 = time.time()
    for i0 in range(0, n, batch):
        idx = range(i0, min(n, i0 + batch))
        depth = predict([seq.image(i) for i in idx])
        for j, dm in zip(idx, depth):
            arr[j] = cv2.resize(np.ascontiguousarray(dm, dtype=np.float32), (wd, hd), interpolation=cv2.INTER_AREA)
        if (i0 // batch) % 50 == 0:
            print(f"  {i0 + len(idx)}/{n}  {(i0 + len(idx)) / (time.time() - t0):.1f} fps")
    arr.flush()
    del arr
    part.rename(out_path)  # only a finished file gets the final name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max_frames", type=int, default=None, help="for a quick test")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}, model {args.model}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    predict = load_model(args.model, device)

    for s in args.seqs:
        path = out_dir / f"depth_{s}.npy"
        if path.exists() and not args.force:
            print(f"seq {s}: already cached, skipping")
            continue
        print(f"seq {s}")
        seq = KittiSequence(args.data_root, s)
        cache_sequence(seq, predict, path, batch=args.batch, max_frames=args.max_frames)
        print(f"  saved {path}")


if __name__ == "__main__":
    main()
