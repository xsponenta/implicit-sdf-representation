# Implicit SDF Representation

Homework project for representing 3D geometry as compact signed distance functions (SDFs). The implementation trains one independent neural SDF per mesh from the provided `.obj` dataset and exports each object as a sub-1 MB `.npz` model.

## Task Requirements

The homework asks for an effective SDF representation for multiple mesh objects with:

| Requirement | Target |
| --- | ---: |
| Single-point SDF retrieval | few nanoseconds |
| Batch retrieval, thousands of points | few milliseconds |
| Serialized memory per object | < 1 MB |
| Occupancy F1 near surface, noise std = `1e-2` | > 0.90 average |
| Occupancy F1 in object bounding volume | > 0.95 average |

The submitted approach is a multi-resolution hash-grid SDF inspired by Instant-NGP, modified for very small memory and very fast CPU inference.

## Method

Each mesh is represented by:

- a multi-resolution spatial hash grid with fp16 feature tables;
- trilinear interpolation over eight hashed voxel corners per level;
- a tiny decoder, `Linear(L*F -> 16) -> ReLU -> Linear(16 -> 1)`;
- a numba inference path for low-overhead single-point and batched queries.

Important changes from a direct Instant-NGP implementation:

- reduced hash-grid size to fit comfortably below 1 MB per object;
- one 16-wide hidden layer instead of a larger MLP;
- online point resampling during training to avoid cached-sample memorization;
- no eikonal/BCE auxiliary losses after ablation showed they hurt this dataset;
- power-of-two hash tables so modulo becomes a bitmask in the inference kernel;
- adaptive table size for harder meshes.

## Results

Saved model files are in `models/`. Every model is below 1 MB; most are about 235 KB and harder meshes are about 469 KB.

Recorded benchmark summaries:

- `models/eval.csv` - occupancy F1 and model size per mesh.
- `models/bench.csv` - torch/numba parity and inference latency per mesh.

Representative inference speed from `bench.csv`:

| Query mode | Typical latency |
| --- | ---: |
| Single point | about 100 ns |
| 1k points | about 0.10 ms |
| 10k points | about 0.31-0.37 ms |
| 100k points | about 2.4-2.9 ms |

## Repository Layout

| Path | Contents |
| --- | --- |
| `sdf_model.py` | PyTorch training model and `.npz` export/import helpers |
| `sdf_inference.py` | numba-accelerated runtime SDF evaluator |
| `train.ipynb` | training, evaluation, and benchmarking notebook |
| `requirements.txt` | Python dependencies |
| `test_task_meshes/` | provided `.obj` meshes |
| `models/` | exported per-object SDF archives plus CSV metrics |
| `data/` | small cached/sample data used by the notebook |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The project uses:

```text
numpy<2.4
pandas
matplotlib
trimesh
rtree
jupyter
ipykernel
scipy
torch
numba
scikit-image
pysdf
```

## Quick Inference

```python
import numpy as np
from sdf_inference import SDFInference

sdf = SDFInference("models/0.npz")
sdf.warmup()

value = sdf.query_one(0.1, 0.2, 0.3)
points = np.random.uniform(-1, 1, size=(1000, 3)).astype("float32")
values = sdf(points)

print(value, values.shape)
```

## Reproduce Experiments

Open `train.ipynb` and run the cells in order. The notebook:

1. loads and normalizes each `.obj` mesh;
2. samples surface, near-surface, and uniform points;
3. trains one SDF model per mesh;
4. exports compressed `.npz` model files;
5. evaluates occupancy F1 and benchmarks inference speed.
