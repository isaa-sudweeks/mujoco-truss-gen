#!/usr/bin/env bash
#SBATCH --job-name=mjx-warp-profile
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --qos=standby
#SBATCH --requeue
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -Eeuo pipefail

PROFILE_REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROFILE_REPO_ROOT}"

if [[ ! -f pyproject.toml || ! -f experiments/benchmark_mjx_env.py ]]; then
    echo "Submit this job from the mujoco-truss-gen repository root." >&2
    exit 2
fi

if command -v uv >/dev/null 2>&1; then
    PROFILE_UV="$(command -v uv)"
elif [[ -x "${HOME}/.local/bin/uv" ]]; then
    PROFILE_UV="${HOME}/.local/bin/uv"
else
    echo "uv was not found. Install uv or add it to PATH before calling sbatch." >&2
    exit 2
fi

PROFILE_JOB_ID="${SLURM_JOB_ID:-manual}"
PROFILE_RESULT_ROOT="${MJX_PROFILE_RESULT_ROOT:-benchmark_results/orc_issue10_${PROFILE_JOB_ID}}"
PROFILE_JAX_VERSION="${MJX_PROFILE_JAX_VERSION:-0.10.1}"
PROFILE_JAX_CUDA_EXTRA="${MJX_PROFILE_JAX_CUDA_EXTRA:-cuda13}"
PROFILE_MARKER_DIR="${PROFILE_RESULT_ROOT}/.stages"

mkdir -p "${PROFILE_RESULT_ROOT}" "${PROFILE_MARKER_DIR}"

profile_stage_done() {
    [[ -f "${PROFILE_MARKER_DIR}/$1.done" ]]
}

profile_mark_done() {
    touch "${PROFILE_MARKER_DIR}/$1.done"
}

profile_python() {
    PYTHONPATH=. "${PROFILE_UV}" run --no-sync python "$@"
}

profile_diagnostic_passes() {
    profile_python -c '
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
summaries = payload["summaries"]
warp_names = [name for name in summaries if "capacity_x" in name]
backend_errors = [name for name, result in summaries.items() if result.get("status") != "ok"]
nonfinite = [name for name in warp_names if summaries[name].get("first_nonfinite") is not None]
capacity_differences = {
    name: value
    for name, value in payload["comparisons"].items()
    if name.endswith("capacity_x1_vs_x2") and value is not None
}
print(json.dumps({
    "backend_errors": backend_errors,
    "nonfinite_warp_runs": nonfinite,
    "capacity_differences": capacity_differences,
}, indent=2, sort_keys=True))
raise SystemExit(0 if not backend_errors and not nonfinite and not capacity_differences else 1)
' "$1"
}

profile_benchmark_passes() {
    profile_python -c '
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
analysis = payload["analysis"]
result = {
    "capacity_gate_passed": analysis["capacity_gate_passed"],
    "required_model_finiteness_passed": analysis["required_model_finiteness"]["passed"],
    "required_model_failures": analysis["required_model_finiteness"]["failures"],
}
print(json.dumps(result, indent=2, sort_keys=True))
raise SystemExit(
    0
    if result["capacity_gate_passed"] and result["required_model_finiteness_passed"]
    else 1
)
' "$1"
}

echo "MJX-Warp profiling job ${PROFILE_JOB_ID}"
echo "Repository: ${PROFILE_REPO_ROOT}"
echo "Artifacts: ${PROFILE_RESULT_ROOT}"
echo "Started: $(date --iso-8601=seconds)"

if ! profile_stage_done metadata; then
    {
        echo "job_id=${PROFILE_JOB_ID}"
        echo "host=$(hostname)"
        echo "started=$(date --iso-8601=seconds)"
        echo "commit=$(git rev-parse HEAD)"
        echo "branch=$(git branch --show-current)"
        echo "uv=${PROFILE_UV}"
        echo "jax_cuda_extra=${PROFILE_JAX_CUDA_EXTRA}"
        echo "jax_version=${PROFILE_JAX_VERSION}"
        git status --short --branch
    } > "${PROFILE_RESULT_ROOT}/job_metadata.txt"
    nvidia-smi > "${PROFILE_RESULT_ROOT}/nvidia_smi.txt"
    profile_mark_done metadata
fi

if ! profile_stage_done environment; then
    if [[ "${MJX_PROFILE_SKIP_ENV_SETUP:-0}" != "1" ]]; then
        "${PROFILE_UV}" sync --frozen --extra dev --extra warp
        "${PROFILE_UV}" pip install \
            --python .venv/bin/python \
            --reinstall \
            "jax[${PROFILE_JAX_CUDA_EXTRA}]==${PROFILE_JAX_VERSION}"
    fi
    "${PROFILE_UV}" pip list --python .venv/bin/python \
        > "${PROFILE_RESULT_ROOT}/python_packages.txt"
    profile_mark_done environment
fi

if ! profile_stage_done cuda_health; then
    profile_python -c '
import jax
import jax.numpy as jnp

devices = jax.devices()
assert devices and devices[0].platform in {"cuda", "gpu"}, devices
matrix = jnp.asarray([[4.0, 1.0], [1.0, 3.0]])
rhs = jnp.asarray([1.0, 2.0])
solution = jnp.linalg.solve(matrix, rhs)
solution.block_until_ready()
print("jax", jax.__version__)
print("backend", jax.default_backend())
print("devices", devices)
print("cusolver_solution", solution)
' > "${PROFILE_RESULT_ROOT}/cuda_health.txt" 2>&1
    cat "${PROFILE_RESULT_ROOT}/cuda_health.txt"
    profile_mark_done cuda_health
fi

if ! profile_stage_done cuda_tests; then
    PYTHONPATH=. "${PROFILE_UV}" run --no-sync pytest \
        -m cuda tests/test_mjx_env.py -v \
        > "${PROFILE_RESULT_ROOT}/cuda_tests.txt" 2>&1
    cat "${PROFILE_RESULT_ROOT}/cuda_tests.txt"
    profile_mark_done cuda_tests
fi

DIAGNOSTIC_8_JSON="${PROFILE_RESULT_ROOT}/realistic_diagnostic_batch8.json"
DIAGNOSTIC_8_TRACES="${PROFILE_RESULT_ROOT}/realistic_diagnostic_batch8_traces.npz"
if ! profile_stage_done diagnostic_batch8; then
    profile_python -m experiments.diagnose_mjx_warp_realistic \
        --batch-size 8 \
        --steps 100 \
        --output "${DIAGNOSTIC_8_JSON}" \
        --traces "${DIAGNOSTIC_8_TRACES}"
    profile_mark_done diagnostic_batch8
fi

if ! profile_diagnostic_passes "${DIAGNOSTIC_8_JSON}" \
    > "${PROFILE_RESULT_ROOT}/diagnostic_batch8_gate.txt" 2>&1; then
    cat "${PROFILE_RESULT_ROOT}/diagnostic_batch8_gate.txt"
    echo "STOPPED_AT=diagnostic_batch8" > "${PROFILE_RESULT_ROOT}/profiling_status.txt"
    echo "The batch-8 realistic diagnostic failed. Full profiling was intentionally skipped."
    exit 0
fi
cat "${PROFILE_RESULT_ROOT}/diagnostic_batch8_gate.txt"

DIAGNOSTIC_128_JSON="${PROFILE_RESULT_ROOT}/realistic_diagnostic_batch128.json"
DIAGNOSTIC_128_TRACES="${PROFILE_RESULT_ROOT}/realistic_diagnostic_batch128_traces.npz"
if ! profile_stage_done diagnostic_batch128; then
    profile_python -m experiments.diagnose_mjx_warp_realistic \
        --batch-size 128 \
        --steps 100 \
        --output "${DIAGNOSTIC_128_JSON}" \
        --traces "${DIAGNOSTIC_128_TRACES}"
    profile_mark_done diagnostic_batch128
fi

if ! profile_diagnostic_passes "${DIAGNOSTIC_128_JSON}" \
    > "${PROFILE_RESULT_ROOT}/diagnostic_batch128_gate.txt" 2>&1; then
    cat "${PROFILE_RESULT_ROOT}/diagnostic_batch128_gate.txt"
    echo "STOPPED_AT=diagnostic_batch128" > "${PROFILE_RESULT_ROOT}/profiling_status.txt"
    echo "The batch-128 realistic diagnostic failed. Benchmark reruns were intentionally skipped."
    exit 0
fi
cat "${PROFILE_RESULT_ROOT}/diagnostic_batch128_gate.txt"

REALISTIC_JSON="${PROFILE_RESULT_ROOT}/mjx_warp_realistic_rerun.json"
REALISTIC_REPORT="${PROFILE_RESULT_ROOT}/mjx_warp_realistic_rerun.md"
if ! profile_stage_done realistic_benchmark; then
    profile_python experiments/benchmark_mjx_env.py \
        --models octahedron:realistic \
        --batch-sizes 128,256,512 \
        --output "${REALISTIC_JSON}" \
        --report "${REALISTIC_REPORT}"
    profile_mark_done realistic_benchmark
fi

if ! profile_benchmark_passes "${REALISTIC_JSON}" \
    > "${PROFILE_RESULT_ROOT}/realistic_benchmark_gate.txt" 2>&1; then
    cat "${PROFILE_RESULT_ROOT}/realistic_benchmark_gate.txt"
    echo "STOPPED_AT=realistic_benchmark" > "${PROFILE_RESULT_ROOT}/profiling_status.txt"
    echo "The realistic benchmark gate failed. The complete matrix was intentionally skipped."
    exit 0
fi
cat "${PROFILE_RESULT_ROOT}/realistic_benchmark_gate.txt"

FULL_JSON="${PROFILE_RESULT_ROOT}/mjx_warp_full_matrix.json"
FULL_REPORT="${PROFILE_RESULT_ROOT}/mjx_warp_full_matrix.md"
if ! profile_stage_done full_benchmark; then
    profile_python experiments/benchmark_mjx_env.py \
        --output "${FULL_JSON}" \
        --report "${FULL_REPORT}"
    profile_mark_done full_benchmark
fi

if profile_benchmark_passes "${FULL_JSON}" \
    > "${PROFILE_RESULT_ROOT}/full_benchmark_gate.txt" 2>&1; then
    echo "FULL_GATE_PASSED=1" > "${PROFILE_RESULT_ROOT}/profiling_status.txt"
else
    echo "FULL_GATE_PASSED=0" > "${PROFILE_RESULT_ROOT}/profiling_status.txt"
fi
cat "${PROFILE_RESULT_ROOT}/full_benchmark_gate.txt"
echo "COMPLETED=1" >> "${PROFILE_RESULT_ROOT}/profiling_status.txt"
echo "Completed: $(date --iso-8601=seconds)"
echo "Artifacts: ${PROFILE_RESULT_ROOT}"
