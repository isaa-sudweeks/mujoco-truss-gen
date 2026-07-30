# MJX-Warp ORC Benchmark

## Adoption status

**Pending.** The candidate implementation has not yet been benchmarked on the
ORC NVIDIA GPU. MJX-JAX remains the default and no Warp adoption claim should
be made from local CPU tests.

The adoption gate requires:

- At least 1.5x aggregate steady-state environment throughput for 1,536
  environments: 512 each of `tetrahedron`, `octahedron`, and
  `henneberg_n6_1tube_2`.
- Passing CUDA parity tests for observations, controls, rewards, termination,
  selective reset, and runtime domain randomization.
- Finite rollouts with no contact or constraint capacity exhaustion.
- Equivalent ten-step results when Warp contact and constraint capacities are
  doubled.

## ORC run

Create a standard virtual environment in the ORC checkout and install the
development, Warp, and CUDA-enabled JAX dependencies. The command below uses
the CUDA 12 JAX wheels, which bundle CUDA and cuDNN user-space libraries but
still require a compatible NVIDIA driver:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,warp]" "jax[cuda12]"
```

If the ORC allocation uses a CUDA 13-compatible driver instead, replace
`jax[cuda12]` with `jax[cuda13]`. Reuse an existing `.venv` by activating it
and rerunning only the final two commands. The benchmark records the resolved
package versions in its JSON output because this `pip` path does not consume
`uv.lock`.

Run these commands inside an allocated GPU job, then confirm that JAX sees the
requested NVIDIA GPU:

```bash
.venv/bin/python -c "import jax; print(jax.default_backend(), jax.devices())"
```

Run the CUDA behavior gates:

```bash
.venv/bin/python -m pytest -m cuda tests/test_mjx_env.py -v
```

Run the full isolated benchmark matrix:

```bash
.venv/bin/python experiments/benchmark_mjx_env.py
```

The benchmark runs MJX-JAX plus all three Warp CUDA graph modes for abstract
`tetrahedron`, `octahedron`, and `henneberg_n6_1tube_2`, and realistic
`octahedron`, at batch sizes 128, 256, and 512. Each matrix cell runs in a fresh
process to isolate compilation caches and GPU memory.

The command replaces this file with the measured report and writes the complete
machine-readable record to `benchmark_results/mjx_warp_orc.json`. That JSON
contains package versions, GPU identity and memory readings, model dimensions,
solver settings, compilation and first-capture costs, steady-state throughput,
latency outliers, full and partial reset costs, and buffer high-water marks.

## Total-speed estimate

This project is intentionally not modifying or re-profiling GNN-SAC. The report
therefore estimates, rather than measures, total training speedup using:

```text
estimated_total_speedup =
    1 / ((1 - 0.4067) + 0.4067 / measured_physics_speedup)
```

The 40.67% value is the previously measured environment-step share of the
post-replay-optimization GNN-SAC hot path. A 1.5x physics improvement predicts
approximately 1.16x total throughput; a 2x physics improvement predicts
approximately 1.26x. These estimates are not end-to-end training benchmarks.
