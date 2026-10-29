# EdgeFusion-VIO

Reliability-aware visual odometry. Week 1: classical monocular VO baseline on KITTI with an evo eval harness.

Illustrative target numbers from planning (0.18 m ATE EuRoC, 4.3% KITTI drift, 16% fewer failures, 81% degraded sequence detection) are targets, not results. Do not cite them until reproduced.

## Run

```bash
pip install -r requirements.txt
python -m src.run_vo --config configs/kitti_00_orb.yaml
python -m src.evaluate --run results/kitti_00_orb
```

Expected data layout under `data_root`:

```
sequences/00/image_0/000000.png ...
sequences/00/calib.txt
poses/00.txt
```

## Layout

- `src/kitti.py` loader
- `src/vo.py` ORB + essential matrix VO
- `src/run_vo.py` runs a config, writes trajectory and per-frame log
- `src/evaluate.py` ATE, RPE, drift, plot, one row per run in `results/results_table.csv`
- `configs/` one yaml per experiment
