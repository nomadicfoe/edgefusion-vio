# EdgeFusion-VIO

Reliability-aware visual odometry. Week 1: classical monocular VO baseline on KITTI with an evo eval harness.

Illustrative target numbers from planning (0.18 m ATE EuRoC, 4.3% KITTI drift, 16% fewer failures, 81% degraded sequence detection) are targets, not results. Do not cite them until reproduced.

## Run

```bash
pip install -r requirements.txt
python -m src.run_vo --config configs/kitti_00_orb.yaml    # baseline: frame-to-frame two-view
python -m src.evaluate --run results/kitti_00_orb
python -m src.run_vo --config configs/kitti_00_pnp.yaml    # landmark map + PnP RANSAC
python -m src.evaluate --run results/kitti_00_pnp
```

Both use the ground-truth step for scale for now (`use_gt_scale` / `scale_source`), so they are comparable. Depth-based scale replaces it later.

Expected data layout under `data_root`:

```
sequences/00/image_0/000000.png ...
sequences/00/calib.txt
poses/00.txt
```

## Layout

- `src/kitti.py` loader
- `src/vo.py` ORB + essential matrix VO, two-view tracker (baseline)
- `src/vo_pnp.py` landmark map, triangulation, PnP RANSAC tracker
- `src/kitti_metric.py` official KITTI translation (%) and rotation (deg/100m) error
- `src/run_vo.py` runs a config, writes trajectory and per-frame log
- `src/evaluate.py` ATE, RPE, RPE drift proxy, KITTI metric, plot, one row per run in `results/results_table.csv`
- `src/make_labels.py` per-frame local error labels (vs GT) and GT-free features, one dataset CSV for the reliability model
- `src/train_reliability.py` gradient boosting reliability model, leave-one-sequence-out, compared with one-feature rules
- `configs/` one yaml per experiment
