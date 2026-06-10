import argparse
from pathlib import Path

import numpy as np

from sdf_inference import SDFInference


def main() -> None:
    parser = argparse.ArgumentParser(description="Query an exported implicit SDF model.")
    parser.add_argument("model", type=Path, help="Path to a models/<id>.npz file.")
    parser.add_argument("--points", type=int, default=1000, help="Number of random points to query.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the demo points.")
    args = parser.parse_args()

    sdf = SDFInference(args.model)
    sdf.warmup()

    rng = np.random.default_rng(args.seed)
    pts = rng.uniform(-1.0, 1.0, size=(args.points, 3)).astype(np.float32)
    values = sdf(pts)

    print(f"model: {args.model}")
    print(f"queried points: {args.points}")
    print(f"sdf min/mean/max: {values.min():.6f} / {values.mean():.6f} / {values.max():.6f}")
    print(f"single point latency: {sdf.bench_single_point():.2f} ns")


if __name__ == "__main__":
    main()
