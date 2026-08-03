# Benchmark results

`experiments/benchmark_mjx_env.py` writes the raw ORC MJX-JAX/MJX-Warp result to
`mjx_warp_orc.json`. Do not add a fabricated or local CPU substitute: the
adoption decision requires the complete NVIDIA GPU matrix and CUDA parity gates.

`experiments/diagnose_mjx_warp_realistic.py` writes a focused diagnostic summary
to `mjx_warp_realistic_diagnostic.json` and full per-step arrays to
`mjx_warp_realistic_diagnostic_traces.npz`. Keep both artifacts together: the
JSON is the reviewable diagnosis and the NPZ is the evidence needed for deeper
field-level inspection.
