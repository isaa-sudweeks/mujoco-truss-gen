# ORC Slurm profiling

Submit the complete issue #10 MJX-Warp profiling workflow from the repository
root:

```bash
sbatch slurm/profile_mjx_warp.sbatch
```

The job requests one GPU, five CPU cores, 32 GB RAM, and the ORC `standby` QoS.
It performs these stages in order:

1. Record the development/Warp environment prepared on the login node.
2. Verify GPU discovery and a cuSolver operation.
3. Run the CUDA-marked MJX behavior tests.
4. Run the realistic-octahedron diagnostic at batches 8 and 128.
5. If both diagnostics pass, rerun the realistic 128/256/512 benchmark.
6. If the realistic finiteness and capacity gates pass, run the complete matrix.

Scientific gate failures stop later expensive stages but leave the Slurm job in
a successful state. Infrastructure, dependency, CUDA, test, or subprocess
failures leave the job failed. Results are written to
`benchmark_results/orc_issue10_<job-id>/`, and `profiling_status.txt` records the
last completed decision point.

The script is requeue-safe: completed stages have markers under the result
directory and are skipped when the same Slurm job restarts after preemption.

ORC compute nodes cannot reach PyPI, so package installation is disabled by
default. Prepare or repair the environment on the login node before submission:

```bash
uv sync --frozen --extra dev --extra warp
uv pip install --python .venv/bin/python --reinstall "jax[cuda13]==0.10.1"
sbatch slurm/profile_mjx_warp.sbatch
```

`MJX_PROFILE_SETUP_ENV=1` is available only for clusters whose compute nodes
have package-index access. Optional overrides include `MJX_PROFILE_JAX_VERSION`,
`MJX_PROFILE_JAX_CUDA_EXTRA`, and `MJX_PROFILE_RESULT_ROOT`.
