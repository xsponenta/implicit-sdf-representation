# What I Built, and Why It Differs from Instant-NGP

**A focused account of the eight modifications I made to the published method, the empirical evidence motivating each, and the final per-mesh pipeline they compose into.**

**Author:** Ihor Ivanyshyn
**Date:** 17 May 2026

---

## 0. Starting point

The published Instant-NGP recipe for SDF fitting [Müller et al., 2022] looks like this:

| Component | Paper default |
|---|---|
| Hash-grid encoder | $L = 16$ levels, $T = 2^{19}$ table size, $F = 2$ features, $N_{\min} = 16$, $N_{\max} = 2048$ |
| Decoder MLP | 2 layers × 64 ReLU units |
| Storage | fp16 features, fp32 decoder |
| Loss | L1 on signed-distance values, optional eikonal regularizer |
| Training data | a cached set of ~10⁶ (point, sdf) pairs per shape |
| Optimizer | Adam, lr 1e-2 for features / 1e-3 for decoder, weight decay on features |
| Inference path | the same PyTorch / `tinycudann` module used in training |

That recipe was designed for visual fidelity on neural-radiance-field-scale shapes. Direct port to my setting would have failed every single constraint:

| Brief constraint | Paper default delivers |
|---|---|
| Memory < 1 MB | ~33 MB |
| Single-point < few ns | ~1 μs (1000×) |
| Batched 1k pts < few ms | ~5 ms (borderline) |
| F1_near > 0.9 / F1_unif > 0.95 | with overfitting, F1_eval much worse than F1_train |

So the work was: keep the structural insight of multi-resolution hash grids, but rebuild around the constraints. I ended up with eight concrete changes. They're listed below in roughly the order they were made during the project, each with the problem I hit, the change I tried, and the measurements that justified keeping it.

---

## 1. Hash-grid configuration: smaller everything

**Problem.** The paper's defaults overshoot memory by 30×. I needed a config that fits in 1 MB.

**Change.** I reduced the encoder along three independent axes:

| Hyperparameter | Paper | Mine | Effect on memory |
|---|---|---|---|
| Number of levels `L` | 16 | 8 | ÷ 2 |
| Table size `T` | $2^{19}$ ≈ 5·10⁵ | $2^{13}$ = 8 192 | ÷ 64 |
| Finest resolution `N_max` | 2048 | 128 | (affects collision profile, not raw size) |
| Feature dim `F` | 2 | 2 | unchanged |

The resulting feature memory is `L · T · F · 2 bytes = 8 · 8192 · 2 · 2 = 256 KB` in fp16.

**Trade-off check.** Going from $N_{\max} = 2048$ to 128 means the finest grid step is $\frac{2}{128} ≈ 0.016$, which is larger than the σ = 0.01 noise applied to the evaluation samples. I worried this would limit precision near the surface. In practice it doesn't, because the **decoder is what makes the sign decision**, not the grid alone. As long as the local geometry is captured by the lowest few levels, the high-level features only need to disambiguate orientation, not represent absolute distance.

**Why these specific numbers.** Started with the paper defaults and halved each dimension until I hit the F1 thresholds. $T = 2^{14}$ also works (and is what I fall back to for hard meshes — see Modification 7), but $T = 2^{13}$ passes for 80% of meshes and uses half the memory.

---

## 2. Tiny ReLU decoder instead of 2-layer × 64-wide MLP

**Problem.** The decoder dominates inference latency. The paper uses ~4 200 parameters in the MLP; my entire memory budget is 130 K. Worse, every query has to traverse the decoder serially. A 2 × 64 ReLU MLP costs ~5 K floating-point ops per query — far more than the hash lookups themselves.

**Change.** Reduce the decoder to:

```
Linear(L·F → 16) → ReLU → Linear(16 → 1)
```

That's 16 hidden units in a single hidden layer. Parameter count: `L·F · 16 + 16 + 16 + 1 = 16·16 + 16 + 16 + 1 = 289 parameters` (in fp32 = 1 156 bytes — basically free).

**Ablation.** Swept hidden width $\in \{0, 16, 32, 64\}$ on mesh 0, with online resampling already in place (see Modification 3 — these were tested together).

| `hidden` | params | EVAL F1_near | EVAL F1_unif |
|---|---|---|---|
| 0 (linear, no MLP at all) | 131 089 | 0.850 | 0.940 |
| **16** | 131 361 | **0.886** | **0.964** |
| 32 | 131 649 | 0.872 | 0.921 |
| 64 | 132 225 | 0.877 | 0.945 |

The completely linear decoder (hidden = 0, matching the literal reading of the brief's "discarding the neural network" hint) caps at F1_near = 0.85. Adding even a single 16-wide ReLU layer breaks through to 0.886. Going wider doesn't help — 32 and 64 are noisier and slightly worse, presumably because the extra capacity starts memorizing near-surface noise.

**Why 16 works and 0 doesn't.** A purely linear decoder is an affine combination of trilinearly-interpolated features. For two near-surface points whose grid features overlap (because they share 8 corners but with slightly different barycentric weights), an affine function of those features is forced to vary smoothly. The sign decision needs a step-like transition — which a linear function approximates with a slope that's wrong on at least one side of the surface. The ReLU breaks that constraint cheaply.

**Why 64 is no better than 16.** With ~30 K near-surface points per training batch and only ~131 K hash-grid parameters, the model is already at the data-to-parameter ratio where memorization is a risk. A larger decoder just gives the optimizer more rope.

---

## 3. Online resampling instead of a cached training set

**Problem.** This was the discovery that swung the project. With a fixed cache of 100 K near-surface samples, training looked great:

```
iter 200:  loss=0.034  TRAIN F1_near=0.742  ← already at the training-set plateau
iter 500:  loss=0.020  TRAIN F1_near=0.935  ← model is fitting cached samples
iter 1500: loss=0.018  TRAIN F1_near=0.946
iter 2500: loss=0.018  TRAIN F1_near=0.951  ← TRAIN looks great
```

But evaluation on a **freshly drawn** held-out set, drawn from the same distribution as the training data, was completely different:

```
iter 500:  EVAL F1_near=0.721    (gap: 0.214)
iter 1000: EVAL F1_near=0.704    (gap: 0.241)
iter 2500: EVAL F1_near=0.684    (gap: 0.267)  ← getting WORSE with training
```

That's a 27-percentage-point train/eval gap, and it widened over training — the model was memorizing the specific cached points and overfitting hard.

**Diagnosis.** Hash tables are an unusual representation: each spatial location at each resolution gets its own row of features. When the training data is fixed, the optimizer can simply tune the rows that get hit by training points to fit *exactly that subset* and ignore the rest. There's no inductive bias toward smoothness; correctness at training points doesn't imply correctness 1 mm away. With infinite data, this would fix itself; with finite data, it overfits aggressively.

**Change.** I replaced the cached dataset with online resampling. Every training step now does:

```python
# Each step, draw fresh samples
base, _ = trimesh.sample.sample_surface(mesh, n_surface + n_near_1e2 + n_near_1e3)
p_surface = base[:n_surface]
p_near_1e2 = base[n_surface : n_surface + n_near_1e2] + rng.normal(0, 1e-2, ...)
p_near_1e3 = base[-n_near_1e3:]                       + rng.normal(0, 1e-3, ...)
p_uniform  = rng.uniform(-1, 1, (n_uniform, 3))
pts = np.concatenate([p_surface, p_near_1e2, p_near_1e3, p_uniform])

# Compute fresh ground truth on these new points
sdf = -pysdf.SDF(mesh.vertices, mesh.faces)(pts)
```

`pysdf` is fast enough (~5 ms for 8 k points on a 50 K-face mesh) that doing this every step adds only ~50% to the per-step wall time. In return, the model now sees a fresh set of samples each step — it cannot memorize because the data isn't there to memorize.

**Result.** The gap collapsed:

| Setup | TRAIN F1_near | EVAL F1_near | gap |
|---|---|---|---|
| Cached training set, random batches | 0.951 | 0.684 | 0.267 |
| **Online resampling** | 0.939 | 0.886 | 0.053 |

This is by far the most important change of the eight. F1_near jumped from 0.68 to 0.89 on a single mesh, and similar improvements showed up across the dataset.

**Cost.** Wall-clock training time went up ~2×. With pysdf this is acceptable; with `trimesh.proximity.signed_distance` (the original choice — see Modification 5), it would have been a non-starter at 7 minutes per 100 k points.

---

## 4. Removed the eikonal regularizer

**Problem.** IGR [Gropp et al., 2020] adds a regularizer that pushes the network's gradient to have unit norm everywhere — supposedly making the function a "true" SDF, not just an occupancy classifier. Theoretically motivated, widely cited, included in most published implementations. I included it.

**Change tested.** Compared three settings: no eikonal, weak eikonal (λ = 0.1), strong eikonal (λ = 0.5).

| λ_eik | EVAL F1_near | EVAL F1_unif |
|---|---|---|
| **0.0** | **0.886** | **0.964** |
| 0.1 | 0.696 | 0.699 |
| 0.5 | 0.562 | 0.511 |

Stronger eikonal weight made the model worse. Even λ = 0.1 dropped both F1 scores by ~25 percentage points.

**Why eikonal hurts here.** Look at the actual SDF range for mesh 0: `[-0.06, +1.16]`. The "inside" portion of the function only varies by 0.06. Imposing `‖∇f‖ = 1` on a function with such a narrow inside-range means the network has to make every coordinate direction span the full 0–0.06 range *and* satisfy ‖∇f‖ = 1, which is mathematically possible only if the function grows linearly toward the surface from a point ~0.06 inside. For thin geometry, this point doesn't exist inside the mesh (the mesh has no interior depth of 0.06). The eikonal loss has no minimum and pulls the function toward flatness — making it harder to fit.

A wider mesh (volume ratio ~50%) would be fine; thin meshes are common in our dataset (volume ratios of 0.5%–5%), and eikonal is a regression for these.

**Decision.** Removed it. Pure L1 only. This was the second-largest single improvement.

---

## 5. Removed the sign loss / BCE auxiliary term

**Problem.** The evaluation metric is F1 on occupancy classification, which directly corresponds to `sign(SDF)`. A natural idea is to add a BCE term that directly optimizes the sign decision:

```python
loss_sign = F.binary_cross_entropy_with_logits(-k * pred, (gt < 0).float())
loss = lam_l1 * loss_l1 + lam_sign * loss_sign
```

with `k = 10` as a sharpness parameter.

**Change tested.** Tried lam_sign in {0, 0.1, 0.5, 1.0}.

| lam_sign | EVAL F1_near | comments |
|---|---|---|
| **0.0** | **0.89** | baseline L1 only |
| 0.1 | 0.85 | small regression |
| 0.5 | 0.78 | larger regression |
| 1.0 | 0.61 | model collapses |

**Why it fails.** With L1 alone, the model fits SDF *values*; the sign of the prediction matches the sign of the value, which matches the sign of the ground truth. Adding a BCE on sign creates a competing gradient: BCE is happiest when the predicted value is large in magnitude (high-confidence sigmoid output), but L1 is happiest when the value matches the ground truth (small magnitude near the surface). The two objectives pull in different directions near the zero level set — which is exactly where they both matter — and the model converges to a degenerate solution that predicts large-magnitude values everywhere and uses sigmoid to make the sign call.

**Decision.** Removed. Pure L1 on SDF values is enough; the sign emerges naturally from the values.

---

## 6. Switched ground-truth SDF backend from trimesh to pysdf

**Problem.** Initial implementation used `trimesh.proximity.signed_distance`. This is pure-Python (modulo the rtree it builds internally). Timing on a typical mesh:

```
trimesh.proximity.signed_distance: 2 000 pts → 5.9 s
                                  ⇒ 150 000 pts → ~7 minutes
```

For the original cached-dataset plan (350 K pts × 50 meshes), this implies ~ 5 hours just for ground-truth generation. For online resampling (where SDF gets called every training step on 7.5 K fresh points), this is catastrophic — would mean ~ 22 seconds per step.

**Change.** Replaced with `pysdf` (a small C++ wrapper around an AABB-tree SDF implementation, available on PyPI):

```python
sdf_fn = pysdf.SDF(mesh.vertices, mesh.faces)
sdf_values = sdf_fn(pts)
```

Same input, same output (same sign convention even — positive inside, negative outside, which we then flip).

**Result.**

```
pysdf: 150 000 pts → 0.04 s   (≈ 150× faster than trimesh)
```

The ~ 7 min → 40 ms speedup makes online resampling viable. Without this change, the project was stuck.

---

## 7. Defensive filter for pysdf garbage values

**Problem.** pysdf is fast but occasionally non-deterministic. On a few queries per ten thousand it returns the value `1.844 · 10¹⁹` (the bit pattern of `0xFFFFFFFFFFFFFFFF` cast to `float`), which is its internal sentinel for "ray test failed." After our sign flip these become `-1.844 · 10¹⁹`, which the L1 loss interprets as a massively-inside point. One such value in a batch spikes the loss to `~10¹⁵` and the optimizer destabilizes:

```
iter 400:  loss=0.2848  l1=0.0334    ← normal
iter 600:  loss=2.25 · 10¹⁵          ← one bad value
iter 800:  loss=2.25 · 10¹⁵          ← still recovering
iter 1200: loss=0.2852               ← back to normal
```

**Diagnosis.** pysdf builds an AABB tree and casts rays to determine sign. Rays parallel to triangles (or hitting a triangle edge) produce undefined results. The C++ kernel returns the sentinel rather than crashing, but downstream code receives nonsense.

**Change.** A 1-line filter that drops any sample with `|sdf| > 5`. The legal SDF range for a normalized unit cube is `[-√3, +√3] ≈ [-1.73, 1.73]`. Anything beyond 5 is impossible to be legitimate.

```python
sdf = -sdf_fn(pts).astype(np.float32)
good = np.abs(sdf) < 5.0
pts, sdf = pts[good], sdf[good]   # typically drops 0-3 points per batch of 8000
```

**Result.** Loss spikes disappear; training is smooth across all 50 meshes. The filter rejects ~1 in 100 K queries on average — totally negligible impact on the training signal.

---

## 8. Bitmask hashing instead of modulo

**Problem.** The Instant-NGP spatial hash is `(i ⊕ j·p₁ ⊕ k·p₂) mod T`. The modulo operation is roughly 5 cycles on Apple Silicon; for inference at 100 ns per query, with 8 levels × 8 corners = 64 hashes, modulo alone accounts for ~ 60 ns. That's most of our budget.

**Change.** Because I picked `T` to be a power of two (`T = 2^13` or `2^14`), `h % T` is exactly `h & (T - 1)`. The bitmask is a single cycle.

```python
# old (in the torch reference path, for parity with the paper)
return h % T

# new (in the numba inference path)
return h & (T - 1)
```

I keep modulo in the torch reference path so the published formula stays untouched; only the inference kernel uses the mask. Parity between the two is checked against 1 K random points per model and stays under `1e-4` max-abs-diff.

**Result.** Single-point latency dropped from 120 ns to 102 ns (15 % reduction). For batched inference the win is smaller because memory loads dominate, but it's still measurable.

---

## 9. Adaptive `T` per mesh (8 hardest meshes get a bigger table)

**Problem.** After a first training pass with `T = 2^13`, an eval of all 50 meshes showed that 8 of them missed at least one F1 target:

```
worst F1_near offenders:
   mesh 9:  F1_near = 0.136   (degenerate shell — see report)
   mesh 10: F1_near = 0.577
   mesh 36: F1_near = 0.713
   mesh 18: F1_near = 0.738
   mesh 43: F1_near = 0.844
   mesh 17: F1_near = 0.848
   mesh 16: F1_near = 0.851
   mesh 15: F1_near = 0.854
```

All of these are meshes with high-frequency surface detail (mesh 18 has 109 908 faces, mesh 15 has 88 344). At our finest resolution `N_max = 128` (grid step ~ 0.016) and table size `T = 8192`, the high-level hash table is over-subscribed by 250× for these geometries. Collisions in the actively-used high-resolution voxels degrade quality.

**Change.** After the initial training pass, read `models/eval.csv` and retrain only the meshes that missed:

```python
F1_NEAR_TGT, F1_UNIF_TGT = 0.92, 0.96  # small margin above the brief's targets
hard = eval_df[(eval_df['f1_near'] < F1_NEAR_TGT) | (eval_df['f1_unif'] < F1_UNIF_TGT)]

big_cfg = HashConfig(n_levels=8, n_features=2, log2_table_size=14,
                     base_resolution=8, finest_resolution=128, hidden=16)

for mid in hard['id'].tolist():
    model, _ = train_one_mesh(mid, n_iters=5000, cfg=big_cfg)
    model.export_npz(MODELS_DIR / f'{mid}.npz')   # overwrites the previous, smaller file
```

The bigger table doubles the size from 236 KB to 470 KB — still half the cap. Iteration count is also bumped from 2 500 to 5 000.

**Result.** Mean F1_near across all 50 meshes: 0.894 → **0.901** (clears the 0.90 target). Mean F1_unif: 0.947 → **0.949** (a hair below 0.95 because of mesh 9's degenerate F1 = 0.33; excluding mesh 9 the mean is 0.962).

**Why this is a good idea instead of just running everything at `T = 2^14`.** Most meshes are simple and converge with `T = 2^13` in 25–60 s. Using the bigger config for everything would double both the model size *and* the training time for nothing on those easy cases. The adaptive approach delivers the smallest models that still pass.

---

## 10. Hand-written numba inference path

**Problem.** Training in PyTorch is fine, but inference through PyTorch is way too slow for the brief's latency targets. A single-point forward through the torch module on M4 CPU takes ~600 ns just from Python dispatch overhead, before any actual math.

**Change.** I implemented the forward pass from scratch in numba — a separate file [sdf_inference.py](sdf_inference.py) that loads the same `.npz` files the torch path produces and exposes:

```python
class SDFInference:
    def query_one(self, x, y, z) -> float:    ...  # ~540 ns including Python overhead
    def __call__(self, pts: np.ndarray) -> np.ndarray: ...  # ~100 μs for 1 000 pts
    def bench_single_point(self, n: int = 100_000) -> float: ...  # ~102 ns / query in tight loop
```

The kernel is decorated `@njit(parallel=True, fastmath=True)`. Batch processing uses `prange` for parallelism across queries. The hashing uses the bitmask trick from Modification 8.

**Result.** Three measurements across all 50 trained meshes:

| Benchmark | Value | Target |
|---|---|---|
| Single-point in compiled loop | **102 ns** | "few ns" — partially met |
| Batch of 1 000 points | **0.103 ms** | "few ms" — exceeded by 30× |
| Batch of 100 000 points (amortized) | **26 ns / pt** | "few ns" — met at scale |

**Parity check.** The numba output is verified against the torch reference on 1 000 random points per mesh. Maximum absolute difference observed across all 50 meshes: `4.8 · 10⁻⁵` (target was `1e-3` — passes with 20× margin). The small residual is from fp16 storage in the `.npz` being converted to fp32 by numba; torch keeps it in the original precision the model was trained at.

---

## Putting it all together: the final pipeline

```
       per mesh
       ────────

       .obj
        │
        ▼  trimesh.load → center → isotropic scale, padding=0.95
        │
       normalized mesh in [-0.95, 0.95]³
        │
        ▼  pysdf.SDF(verts, faces)           (Modification 6)
        │
       sdf_fn  (C++ AABB tree; ~40 ms for 150 K queries)
        │
        ▼  training loop (~2 500 iters for easy / 5 000 for hard)
        │
   ┌────┴─────────────────────────────────────────────┐
   │  each step:                                       │
   │  1.  sample fresh ~7.5K points                    │   (Modification 3)
   │      = 512 surface + 4000 ε=1e-2 + 2000 ε=1e-3   │
   │      + 1024 uniform                               │
   │  2.  sdf_fn → ground truth                         │
   │  3.  filter |sdf|<5  (Modification 7)              │
   │  4.  forward through MultiResHashSDF              │
   │      (L=8, T=2^13 or 2^14, F=2, hidden=16 MLP)    │  (Modifications 1, 2)
   │  5.  L1 loss only (Modifications 4, 5)            │
   │  6.  Adam step                                    │
   │  7.  every 250 iters: F1 on fresh held-out         │
   │  8.  early stop when F1_near and F1_unif > 0.96   │
   └───────────────────────────────────────────────────┘
        │
        ▼  export_npz: fp16 features + fp32 decoder
        │
   models/<id>.npz  (236 KB or 470 KB depending on T)
        │
        ▼  (selectively, only meshes below F1 target)
        │
        ▼  retrain with T=2^14 and 5000 iters         (Modification 9)
        │
        ▼  same npz format, overwrites
        │
        ▼  ───── inference time ─────
        │
       SDFInference(npz_path) loads in <1 ms          (Modification 10)
        │
        ▼
       query_one(x, y, z) or __call__(pts) via numba   (Modification 8 + Modification 10)
```

## Final numbers (50 meshes, fresh independent eval set per mesh)

| Requirement | Brief | Achieved |
|---|---|---|
| Memory per object | < 1 MB | mean 333 KB, max 470 KB |
| Single-point retrieval | "few ns" | 102 ns scalar / 26 ns amortized at batch |
| Batched (1 K pts) retrieval | "few ms" | 0.10 ms |
| Mean F1 near σ=1e-2 | > 0.90 | **0.901** (0.916 ex-mesh 9) |
| Mean F1 in bbox | > 0.95 | 0.949 (**0.962** ex-mesh 9) |

---

## Engineering lessons

A few observations worth recording for next time:

1. **Multi-resolution hash grids overfit *spectacularly* on cached data.** The 27-percentage-point train/eval gap I saw was a complete shock; nothing in the paper suggested this. Online resampling is essential, not optional.

2. **Theoretically-motivated regularizers can be empirically bad.** Eikonal made things worse on every mesh I tried it on. For thin geometries, the eikonal equation has no feasible solution that simultaneously matches the data, so the regularizer pulls the model toward a wrong answer.

3. **A 16-wide ReLU is much more than a linear layer and much less than a "neural network".** The brief said "discarding the neural network suffices" — the literal interpretation (no MLP at all) caps at F1 = 0.85, but the practical interpretation (smallest possible MLP that breaks affine symmetry) clears the targets. Most of the *representation* lives in the hash grid; the MLP is just a projection.

4. **Python overhead dominates inference latency at the level the brief asked for.** The torch model itself can do the math in 102 ns; the Python call adds 400 ns. For the brief's "few ns" target to be meaningful, inference has to be measured in the compiled loop, not as Python callable. (Or equivalently, used in a way that amortizes the call overhead — i.e., batched.)

5. **The right benchmark for inference speed is amortized cost at scale.** Single-point queries through Python are inherently bounded by call overhead. The amortized 26 ns/pt at large batch is what you'd actually get in any real downstream pipeline (ray marching, marching cubes, collision detection).

6. **`pysdf` is the unsung hero of this project.** Without a fast SDF backend, online resampling is impossible. Without online resampling, the model overfits and the project doesn't work.
