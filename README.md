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

The baseline and `kitti_00_pnp.yaml` use the ground-truth step for scale, so they are comparable but not real monocular VO. `kitti_00_pnpdepth.yaml` takes scale from cached network depth (`scale_source: depth`) and never reads ground truth in the loop.
Every run writes `run_info.json` with library versions, because results change with the OpenCV version.

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
- `src/extra_features.py` image sharpness/brightness, optical flow, and flow vs estimated-pose epipolar error (no VO rerun needed)
- `src/depth_cache.py` runs a metric depth network once and caches quarter-res depth maps (needs torch and transformers, preinstalled in Colab)
- `src/train_reliability.py` gradient boosting reliability model, leave-one-sequence-out, compared with one-feature rules
- `configs/` one yaml per experiment
