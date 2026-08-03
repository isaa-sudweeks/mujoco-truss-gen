from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import mujoco  # type: ignore[import-untyped]
import numpy as np

from mujoco_truss_gen import MjxNodeVelocityEnv, TrussEnvConfig, get_mujoco_spec
from mujoco_truss_gen.mujoco_model.model import MujocoModel

DEFAULT_MODELS = (
    "tetrahedron:abstract",
    "octahedron:abstract",
    "henneberg_n6_1tube_2:abstract",
    "octahedron:realistic",
)
DEFAULT_BATCH_SIZES = (128, 256, 512)
DEFAULT_WARP_GRAPH_MODES = ("warp", "warp_staged", "warp_staged_ex")
REPRESENTATIVE_MODELS = (
    "tetrahedron:abstract",
    "octahedron:abstract",
    "henneberg_n6_1tube_2:abstract",
)
ENVIRONMENT_HOT_PATH_FRACTION = 0.4067
ADOPTION_SPEEDUP_THRESHOLD = 1.5


@dataclass(frozen=True)
class BenchmarkCase:
    model: str
    batch_size: int
    implementation: Literal["jax", "warp"]
    graph_mode: Literal["warp", "warp_staged", "warp_staged_ex"]
    nsubsteps: int
    seed: int
    warmup_steps: int
    block_count: int
    steps_per_block: int
    latency_samples: int
    reset_samples: int
    warp_contact_capacity_per_env: int | None
    warp_constraint_capacity: int | None
    capacity_scale: int = 1
    capacity_validation_only: bool = False


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _parse_csv(value))


def _parse_model(value: str) -> tuple[str, bool]:
    try:
        preset, kind = value.rsplit(":", 1)
    except ValueError as error:
        raise ValueError(
            f"Model {value!r} must use PRESET:abstract or PRESET:realistic."
        ) from error
    if kind not in {"abstract", "realistic"}:
        raise ValueError(f"Model {value!r} must end in :abstract or :realistic, not :{kind}.")
    return preset, kind == "realistic"


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return math.nan
    return float(np.percentile(np.asarray(values), percentile))


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "median_seconds": math.nan,
            "p95_seconds": math.nan,
            "max_seconds": math.nan,
        }
    return {
        "count": len(values),
        "median_seconds": statistics.median(values),
        "p95_seconds": _percentile(values, 95),
        "max_seconds": max(values),
    }


def _outliers(values: list[float]) -> list[dict[str, float | int]]:
    if len(values) < 4:
        return []
    q1 = _percentile(values, 25)
    q3 = _percentile(values, 75)
    threshold = q3 + 1.5 * (q3 - q1)
    return [
        {"index": index, "seconds": value}
        for index, value in enumerate(values)
        if value > threshold
    ]


def _block_until_ready(tree: Any) -> None:
    leaves = jax.tree.leaves(tree)
    for leaf in leaves:
        block = getattr(leaf, "block_until_ready", None)
        if callable(block):
            block()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _gpu_process_memory_mib() -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None

    memory = 0
    found = False
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2 or fields[0] != str(os.getpid()):
            continue
        memory += int(fields[1])
        found = True
    return memory if found else 0


def _jax_memory_stats() -> dict[str, int] | None:
    stats = jax.devices()[0].memory_stats()
    if not stats:
        return None
    return {
        key: int(value)
        for key, value in stats.items()
        if isinstance(value, (int, np.integer))
        and key in {"bytes_in_use", "peak_bytes_in_use", "bytes_limit"}
    }


def _capacity_defaults(
    model_source: Any,
    batch_size: int,
    *,
    contact_capacity_per_env: int | None,
    constraint_capacity: int | None,
    scale: int,
) -> tuple[int, int, dict[str, int]]:
    native_model = MujocoModel(model_source).model
    per_env_contacts = contact_capacity_per_env or max(16, 4 * int(native_model.ngeom))
    per_env_constraints = constraint_capacity or max(
        128,
        8 * (int(native_model.nv) + int(native_model.neq) + int(native_model.ntendon)),
    )
    capacities = {
        "contact_capacity_per_env": per_env_contacts * scale,
        "contact_capacity_total": per_env_contacts * batch_size * scale,
        "constraint_capacity_per_env": per_env_constraints * scale,
    }
    return (
        capacities["contact_capacity_total"],
        capacities["constraint_capacity_per_env"],
        capacities,
    )


def _timed_call(function: Any, *args: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    result = function(*args)
    _block_until_ready(result)
    return result, time.perf_counter() - start


def _record_buffer_high_water(
    env: MjxNodeVelocityEnv,
    state: Any,
    high_water: dict[str, int | bool | None],
) -> None:
    diagnostics = env.buffer_diagnostics(state)
    for name in ("contact_count", "constraint_count_max"):
        value = diagnostics[name]
        if value is not None:
            high_water[name] = max(int(high_water.get(name) or 0), int(value))
    high_water["overflow"] = bool(high_water.get("overflow")) or bool(diagnostics["overflow"])


def run_case(case: BenchmarkCase) -> dict[str, Any]:
    preset, realistic = _parse_model(case.model)
    model_source = get_mujoco_spec(preset, realistic=realistic)
    warp_naconmax = None
    warp_njmax = None
    capacities: dict[str, int] | None = None
    if case.implementation == "warp":
        warp_naconmax, warp_njmax, capacities = _capacity_defaults(
            model_source,
            case.batch_size,
            contact_capacity_per_env=case.warp_contact_capacity_per_env,
            constraint_capacity=case.warp_constraint_capacity,
            scale=case.capacity_scale,
        )

    memory_before_mib = _gpu_process_memory_mib()
    config = TrussEnvConfig(
        model_source,
        nsubsteps=case.nsubsteps,
        max_steps=100_000,
        runtime_apply_control_noise=False,
    )
    construction_start = time.perf_counter()
    env = MjxNodeVelocityEnv(
        config,
        mjx_impl=case.implementation,
        warp_graph_mode=case.graph_mode,
        warp_naconmax=warp_naconmax,
        warp_njmax=warp_njmax,
    )
    construction_seconds = time.perf_counter() - construction_start

    model = env.mujoco_model.model
    keys = jax.random.split(jax.random.key(case.seed), case.batch_size)
    action_rng = np.random.default_rng(case.seed)
    action_count = max(
        1,
        case.warmup_steps + case.block_count * case.steps_per_block + case.latency_samples,
    )
    actions = jnp.asarray(
        action_rng.uniform(
            low=-float(config.speed),
            high=float(config.speed),
            size=(action_count, case.batch_size, env.action_size),
        ),
        dtype=jnp.float32,
    )

    reset_lower_start = time.perf_counter()
    reset_executable = jax.jit(env.reset).lower(keys).compile()
    reset_compile_seconds = time.perf_counter() - reset_lower_start
    (obs, state), reset_first_execution_seconds = _timed_call(reset_executable, keys)

    step_lower_start = time.perf_counter()
    step_executable = jax.jit(env.step).lower(keys, state, actions[0]).compile()
    step_compile_seconds = time.perf_counter() - step_lower_start
    step_result, first_execution_seconds = _timed_call(step_executable, keys, state, actions[0])
    obs, state, reward, done, info = step_result

    high_water: dict[str, int | bool | None] = {
        "contact_count": 0,
        "constraint_count_max": 0,
        "overflow": False,
    }
    _record_buffer_high_water(env, state, high_water)

    if case.capacity_validation_only:
        validation_steps = max(1, case.latency_samples)
        for step_index in range(validation_steps):
            obs, state, reward, done, info = step_executable(
                keys, state, actions[step_index % action_count]
            )
        _block_until_ready((obs, state, reward, done, info))
        _record_buffer_high_water(env, state, high_water)
        return {
            "status": "ok",
            "case": asdict(case),
            "capacities": capacities,
            "capacity_high_water": high_water,
            "validation": {
                "observation_checksum": float(jnp.sum(obs)),
                "reward_checksum": float(jnp.sum(reward)),
                "done_count": int(jnp.sum(done)),
                "finite": bool(jnp.all(jnp.isfinite(obs)) and jnp.all(jnp.isfinite(reward))),
            },
        }

    for step_index in range(case.warmup_steps):
        obs, state, reward, done, info = step_executable(
            keys, state, actions[step_index % action_count]
        )
    _block_until_ready((obs, state, reward, done, info))
    _record_buffer_high_water(env, state, high_water)

    block_seconds: list[float] = []
    action_index = case.warmup_steps
    for _ in range(case.block_count):
        start = time.perf_counter()
        for _ in range(case.steps_per_block):
            obs, state, reward, done, info = step_executable(
                keys, state, actions[action_index % action_count]
            )
            action_index += 1
        _block_until_ready((obs, state, reward, done, info))
        block_seconds.append(time.perf_counter() - start)
        _record_buffer_high_water(env, state, high_water)

    latency_seconds: list[float] = []
    for _ in range(case.latency_samples):
        start = time.perf_counter()
        obs, state, reward, done, info = step_executable(
            keys, state, actions[action_index % action_count]
        )
        action_index += 1
        _block_until_ready((obs, state, reward, done, info))
        latency_seconds.append(time.perf_counter() - start)
        _record_buffer_high_water(env, state, high_water)

    reset_seconds: list[float] = []
    for _ in range(case.reset_samples):
        (_, _), elapsed = _timed_call(reset_executable, keys)
        reset_seconds.append(elapsed)

    reset_where_compile_start = time.perf_counter()
    ten_percent_mask = jnp.arange(case.batch_size) < max(1, math.ceil(0.10 * case.batch_size))
    reset_where_executable = jax.jit(env.reset_where).lower(keys, state, ten_percent_mask).compile()
    reset_where_compile_seconds = time.perf_counter() - reset_where_compile_start
    partial_reset: dict[str, dict[str, float | int]] = {}
    for fraction in (0.01, 0.10, 0.50):
        reset_count = max(1, math.ceil(fraction * case.batch_size))
        mask = jnp.arange(case.batch_size) < reset_count
        samples: list[float] = []
        partial_state = state
        for _ in range(case.reset_samples):
            (_, partial_state), elapsed = _timed_call(
                reset_where_executable, keys, partial_state, mask
            )
            samples.append(elapsed)
        partial_reset[f"{fraction:.2f}"] = {
            "reset_count": reset_count,
            **_summary(samples),
        }
        _record_buffer_high_water(env, partial_state, high_water)

    total_block_seconds = sum(block_seconds)
    measured_vector_steps = case.block_count * case.steps_per_block
    measured_transitions = measured_vector_steps * case.batch_size
    memory_after_mib = _gpu_process_memory_mib()
    final_diagnostics = env.buffer_diagnostics(state)

    return {
        "status": "ok",
        "case": asdict(case),
        "versions": {
            "python": platform.python_version(),
            "jax": _package_version("jax"),
            "jaxlib": _package_version("jaxlib"),
            "mujoco": mujoco.__version__,
            "mujoco_mjx": _package_version("mujoco-mjx"),
            "warp_lang": _package_version("warp-lang"),
        },
        "device": {
            "platform": jax.devices()[0].platform,
            "kind": jax.devices()[0].device_kind,
            "memory_before_mib": memory_before_mib,
            "memory_after_mib": memory_after_mib,
            "jax_memory": _jax_memory_stats(),
        },
        "model": {
            "preset": preset,
            "realistic": realistic,
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "nbody": int(model.nbody),
            "ngeom": int(model.ngeom),
            "ntendon": int(model.ntendon),
            "nwrap": int(model.nwrap),
            "neq": int(model.neq),
            "nsensor": int(model.nsensor),
            "npair": int(model.npair),
            "nexclude": int(model.nexclude),
            "integrator": int(model.opt.integrator),
            "solver": int(model.opt.solver),
            "cone": int(model.opt.cone),
            "jacobian": int(model.opt.jacobian),
            "disableflags": int(model.opt.disableflags),
            "iterations": int(model.opt.iterations),
            "ls_iterations": int(model.opt.ls_iterations),
            "equality_types": sorted({int(value) for value in model.eq_type}),
            "geom_types": sorted({int(value) for value in model.geom_type}),
            "actuator_transmission_types": sorted({int(value) for value in model.actuator_trntype}),
            "actuator_dynamics_types": sorted({int(value) for value in model.actuator_dyntype}),
            "actuator_gain_types": sorted({int(value) for value in model.actuator_gaintype}),
            "actuator_bias_types": sorted({int(value) for value in model.actuator_biastype}),
            "sensor_types": sorted({int(value) for value in model.sensor_type}),
        },
        "capacities": capacities,
        "timing": {
            "construction_seconds": construction_seconds,
            "reset_compile_seconds": reset_compile_seconds,
            "reset_first_execution_seconds": reset_first_execution_seconds,
            "step_compile_seconds": step_compile_seconds,
            "step_first_execution_seconds": first_execution_seconds,
            "reset_where_compile_seconds": reset_where_compile_seconds,
            "block_seconds": block_seconds,
            "total_block_seconds": total_block_seconds,
            "measured_vector_steps": measured_vector_steps,
            "measured_transitions": measured_transitions,
            "transitions_per_second": measured_transitions / total_block_seconds,
            "physics_substeps_per_second": (
                measured_transitions * case.nsubsteps / total_block_seconds
            ),
            "vector_step_seconds": total_block_seconds / measured_vector_steps,
            "latency": _summary(latency_seconds),
            "latency_outliers": _outliers(latency_seconds),
            "full_reset": _summary(reset_seconds),
            "partial_reset": partial_reset,
        },
        "capacity_high_water": high_water,
        "final_buffer_diagnostics": final_diagnostics,
        "validation": {
            "observation_checksum": float(jnp.sum(obs)),
            "reward_checksum": float(jnp.sum(reward)),
            "done_count": int(jnp.sum(done)),
            "finite": bool(jnp.all(jnp.isfinite(obs)) and jnp.all(jnp.isfinite(reward))),
        },
    }


def _case_key(result: dict[str, Any]) -> tuple[str, int, str, str]:
    case = result["case"]
    return (
        case["model"],
        int(case["batch_size"]),
        case["implementation"],
        case["graph_mode"],
    )


def _capacity_comparisons(
    validation_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base = {}
    doubled_results = []
    comparisons = []
    for result in validation_results:
        if result.get("status") != "ok":
            comparisons.append(
                {
                    "case": result.get("case"),
                    "passed": False,
                    "reason": result.get("error", "capacity validation case failed"),
                }
            )
            continue
        if int(result["case"]["capacity_scale"]) == 1:
            base[_case_key(result)] = result
        elif int(result["case"]["capacity_scale"]) == 2:
            doubled_results.append(result)

    for doubled in doubled_results:
        key = _case_key(doubled)
        original = base.get(key)
        if original is None:
            comparisons.append(
                {
                    "case": doubled["case"],
                    "passed": False,
                    "reason": "matching base-capacity result is unavailable",
                }
            )
            continue
        base_validation = original["validation"]
        doubled_validation = doubled["validation"]
        checksum_close = math.isclose(
            base_validation["observation_checksum"],
            doubled_validation["observation_checksum"],
            rel_tol=1e-5,
            abs_tol=1e-5,
        ) and math.isclose(
            base_validation["reward_checksum"],
            doubled_validation["reward_checksum"],
            rel_tol=1e-5,
            abs_tol=1e-5,
        )
        passed = (
            checksum_close
            and base_validation["done_count"] == doubled_validation["done_count"]
            and bool(base_validation["finite"])
            and bool(doubled_validation["finite"])
            and not bool(original["capacity_high_water"]["overflow"])
            and not bool(doubled["capacity_high_water"]["overflow"])
        )
        comparisons.append(
            {
                "model": key[0],
                "batch_size": key[1],
                "graph_mode": key[3],
                "passed": passed,
                "base_validation": base_validation,
                "doubled_validation": doubled_validation,
            }
        )
    return comparisons


def _aggregate_workloads(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    successful = {_case_key(result): result for result in results if result.get("status") == "ok"}
    jax_times = []
    for model in REPRESENTATIVE_MODELS:
        result = successful.get((model, 512, "jax", "warp"))
        if result is None:
            return []
        jax_times.append(result["timing"]["vector_step_seconds"])
    jax_total = sum(jax_times)
    jax_throughput = 1536 / jax_total

    workloads = []
    for graph_mode in DEFAULT_WARP_GRAPH_MODES:
        warp_results = [
            successful.get((model, 512, "warp", graph_mode)) for model in REPRESENTATIVE_MODELS
        ]
        if any(result is None for result in warp_results):
            continue
        available_warp_results = [result for result in warp_results if result is not None]
        warp_total = sum(
            result["timing"]["vector_step_seconds"] for result in available_warp_results
        )
        warp_throughput = 1536 / warp_total
        physics_speedup = warp_throughput / jax_throughput
        total_estimate = 1 / (
            (1 - ENVIRONMENT_HOT_PATH_FRACTION) + ENVIRONMENT_HOT_PATH_FRACTION / physics_speedup
        )
        workloads.append(
            {
                "graph_mode": graph_mode,
                "jax_transitions_per_second": jax_throughput,
                "warp_transitions_per_second": warp_throughput,
                "physics_speedup": physics_speedup,
                "estimated_total_training_speedup": total_estimate,
                "performance_gate_passed": physics_speedup >= ADOPTION_SPEEDUP_THRESHOLD,
            }
        )
    return workloads


def _required_model_finiteness(
    results: list[dict[str, Any]],
    *,
    required_models: tuple[str, ...],
    batch_sizes: tuple[int, ...],
    implementations: tuple[str, ...],
    graph_modes: tuple[str, ...],
) -> dict[str, Any]:
    """Summarize the benchmark-local required-model behavioral gate.

    A missing or errored required case is not silently treated as passing. JAX
    has one graph-independent case per model/batch; Warp must be finite for
    every requested graph mode.
    """

    successful = {_case_key(result): result for result in results if result.get("status") == "ok"}
    cases: list[dict[str, Any]] = []
    for model in required_models:
        for batch_size in batch_sizes:
            for implementation in implementations:
                modes = ("warp",) if implementation == "jax" else graph_modes
                for graph_mode in modes:
                    key = (model, batch_size, implementation, graph_mode)
                    result = successful.get(key)
                    finite = bool(result and result.get("validation", {}).get("finite"))
                    cases.append(
                        {
                            "model": model,
                            "batch_size": batch_size,
                            "implementation": implementation,
                            "graph_mode": graph_mode,
                            "present": result is not None,
                            "finite": finite,
                            "done_count": (
                                result.get("validation", {}).get("done_count")
                                if result is not None
                                else None
                            ),
                            "passed": result is not None and finite,
                        }
                    )
    return {
        "passed": bool(cases) and all(case["passed"] for case in cases),
        "cases": cases,
        "failures": [case for case in cases if not case["passed"]],
    }


def _format_float(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# MJX-Warp ORC Benchmark",
        "",
        "This report is generated by `experiments/benchmark_mjx_env.py`. "
        "Compilation and first CUDA graph capture are excluded from steady-state throughput.",
        "",
        "## Adoption status",
        "",
    ]
    workloads = payload["analysis"]["representative_workloads"]
    capacity_passed = payload["analysis"]["capacity_gate_passed"]
    finiteness = payload["analysis"]["required_model_finiteness"]
    finiteness_passed = bool(finiteness["passed"])
    if not workloads:
        lines.append(
            "**Pending.** The complete 512-environment JAX/Warp representative workload "
            "has not been measured."
        )
    else:
        best = max(workloads, key=lambda item: item["physics_speedup"])
        performance_passed = bool(best["performance_gate_passed"])
        adoption_ready = performance_passed and capacity_passed and finiteness_passed
        status = (
            "Benchmark-local gates passed; CUDA parity suite still required"
            if adoption_ready
            else "Do not adopt"
        )
        lines.extend(
            [
                f"**{status}.** Best measured graph mode: `{best['graph_mode']}`.",
                "",
                f"- Physics speedup: {_format_float(best['physics_speedup'])}×",
                f"- Estimated total training speedup: "
                f"{_format_float(best['estimated_total_training_speedup'])}×",
                f"- 1.5× performance gate: {'pass' if performance_passed else 'fail'}",
                f"- Capacity gate: {'pass' if capacity_passed else 'fail'}",
                f"- Required-model finiteness gate: {'pass' if finiteness_passed else 'fail'}",
                "- Behavioral parity gate: must also pass the CUDA-marked pytest suite "
                "before public adoption.",
            ]
        )

    if finiteness["failures"]:
        lines.extend(
            [
                "",
                "### Required-model behavioral failures",
                "",
                "| Model | Batch | Implementation | Graph mode | Present | Finite | Done |",
                "|---|---:|---|---|---|---|---:|",
            ]
        )
        for failure in finiteness["failures"]:
            lines.append(
                f"| `{failure['model']}` | {failure['batch_size']} | "
                f"`{failure['implementation']}` | `{failure['graph_mode']}` | "
                f"{'yes' if failure['present'] else 'no'} | "
                f"{'yes' if failure['finite'] else 'no'} | "
                f"{failure['done_count'] if failure['done_count'] is not None else 'n/a'} |"
            )

    lines.extend(
        [
            "",
            "The total-training number is an Amdahl-law estimate using the previously "
            "measured 40.67% environment-step hot-path share. It is not an end-to-end "
            "GNN-SAC measurement.",
            "",
            "## Representative 1,536-environment workload",
            "",
            "| Graph mode | JAX transitions/s | Warp transitions/s | Physics speedup | "
            "Estimated total speedup | Gate |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for workload in workloads:
        lines.append(
            f"| `{workload['graph_mode']}` | "
            f"{workload['jax_transitions_per_second']:.1f} | "
            f"{workload['warp_transitions_per_second']:.1f} | "
            f"{workload['physics_speedup']:.3f}× | "
            f"{workload['estimated_total_training_speedup']:.3f}× | "
            f"{'pass' if workload['performance_gate_passed'] else 'fail'} |"
        )
    if not workloads:
        lines.append("| pending | n/a | n/a | n/a | n/a | pending |")

    lines.extend(
        [
            "",
            "## Per-case steady-state results",
            "",
            "| Model | Kind | Batch | Implementation | Graph mode | Transitions/s | "
            "Median latency (ms) | p95 latency (ms) | Overflow |",
            "|---|---|---:|---|---|---:|---:|---:|---|",
        ]
    )
    for result in payload["results"]:
        case = result.get("case", {})
        if result.get("status") != "ok":
            lines.append(
                f"| `{case.get('model', 'unknown')}` | n/a | "
                f"{case.get('batch_size', 'n/a')} | `{case.get('implementation', 'n/a')}` | "
                f"`{case.get('graph_mode', 'n/a')}` | error | error | error | error |"
            )
            continue
        timing = result["timing"]
        preset, realistic = _parse_model(case["model"])
        lines.append(
            f"| `{preset}` | {'realistic' if realistic else 'abstract'} | "
            f"{case['batch_size']} | `{case['implementation']}` | `{case['graph_mode']}` | "
            f"{timing['transitions_per_second']:.1f} | "
            f"{1000 * timing['latency']['median_seconds']:.3f} | "
            f"{1000 * timing['latency']['p95_seconds']:.3f} | "
            f"{'yes' if result['capacity_high_water']['overflow'] else 'no'} |"
        )

    metadata = payload["metadata"]
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```bash",
            metadata["command"],
            "```",
            "",
            f"- Generated: `{metadata['generated_at']}`",
            f"- Host: `{metadata['host']}`",
            f"- Python: `{metadata['python']}`",
            f"- JAX default backend: `{metadata['jax_backend']}`",
            f"- Device: `{metadata['jax_device']}`",
            "",
            "Raw package versions, model dimensions, solver settings, graph-capture "
            "timings, reset timings, memory readings, outliers, and capacity high-water "
            "marks are retained in the companion JSON file.",
            "",
            "## Focused realistic-model diagnostic",
            "",
            "If any realistic-model case is non-finite or capacity-dependent, run the "
            "seeded diagnostic before repeating the full matrix:",
            "",
            "```bash",
            "python -m experiments.diagnose_mjx_warp_realistic --batch-size 8 --steps 100",
            "```",
            "",
            "It runs equivalent saved initial states and actions through native MuJoCo, "
            "MJX-JAX, and every Warp graph mode at base and doubled capacities. The JSON "
            "summary identifies the first non-finite or divergent field and environment; "
            "the companion NPZ retains full per-step traces.",
            "",
        ]
    )
    return "\n".join(lines)


def _run_subprocess_case(script: Path, case: BenchmarkCase) -> dict[str, Any]:
    command = [
        sys.executable,
        str(script),
        "--_case-json",
        json.dumps(asdict(case), separators=(",", ":")),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        return {
            "status": "error",
            "case": asdict(case),
            "error": completed.stderr.strip() or completed.stdout.strip(),
        }
    for line in reversed(completed.stdout.splitlines()):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {
        "status": "error",
        "case": asdict(case),
        "error": f"Child process produced no JSON result:\n{completed.stdout}",
    }


def _build_cases(args: argparse.Namespace) -> list[BenchmarkCase]:
    cases = []
    for model in args.models:
        _parse_model(model)
        for batch_size in args.batch_sizes:
            if "jax" in args.implementations:
                cases.append(
                    BenchmarkCase(
                        model=model,
                        batch_size=batch_size,
                        implementation="jax",
                        graph_mode="warp",
                        nsubsteps=args.nsubsteps,
                        seed=args.seed,
                        warmup_steps=args.warmup_steps,
                        block_count=args.block_count,
                        steps_per_block=args.steps_per_block,
                        latency_samples=args.latency_samples,
                        reset_samples=args.reset_samples,
                        warp_contact_capacity_per_env=None,
                        warp_constraint_capacity=None,
                    )
                )
            if "warp" in args.implementations:
                for graph_mode in args.warp_graph_modes:
                    cases.append(
                        BenchmarkCase(
                            model=model,
                            batch_size=batch_size,
                            implementation="warp",
                            graph_mode=graph_mode,
                            nsubsteps=args.nsubsteps,
                            seed=args.seed,
                            warmup_steps=args.warmup_steps,
                            block_count=args.block_count,
                            steps_per_block=args.steps_per_block,
                            latency_samples=args.latency_samples,
                            reset_samples=args.reset_samples,
                            warp_contact_capacity_per_env=args.warp_contact_capacity_per_env,
                            warp_constraint_capacity=args.warp_constraint_capacity,
                        )
                    )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark MJX-JAX and MJX-Warp truss environments in fresh processes."
    )
    parser.add_argument("--models", type=_parse_csv, default=DEFAULT_MODELS)
    parser.add_argument("--batch-sizes", type=_parse_int_csv, default=DEFAULT_BATCH_SIZES)
    parser.add_argument(
        "--implementations",
        type=_parse_csv,
        choices=None,
        default=("jax", "warp"),
        help="Comma-separated subset of jax,warp.",
    )
    parser.add_argument(
        "--warp-graph-modes",
        type=_parse_csv,
        default=DEFAULT_WARP_GRAPH_MODES,
    )
    parser.add_argument("--nsubsteps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--block-count", type=int, default=10)
    parser.add_argument("--steps-per-block", type=int, default=20)
    parser.add_argument("--latency-samples", type=int, default=100)
    parser.add_argument("--reset-samples", type=int, default=10)
    parser.add_argument("--warp-contact-capacity-per-env", type=int)
    parser.add_argument("--warp-constraint-capacity", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/mjx_warp_orc.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/benchmarks/mjx_warp.md"),
    )
    parser.add_argument("--_case-json", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._case_json:
        try:
            result = run_case(BenchmarkCase(**json.loads(args._case_json)))
        except Exception as error:
            result = {
                "status": "error",
                "case": json.loads(args._case_json),
                "error": f"{type(error).__name__}: {error}",
            }
            print(json.dumps(result, sort_keys=True))
            raise SystemExit(1) from error
        print(json.dumps(result, sort_keys=True))
        return

    invalid_implementations = set(args.implementations) - {"jax", "warp"}
    if invalid_implementations:
        parser.error(f"Unsupported implementations: {', '.join(sorted(invalid_implementations))}")
    invalid_graph_modes = set(args.warp_graph_modes) - set(DEFAULT_WARP_GRAPH_MODES)
    if invalid_graph_modes:
        parser.error(f"Unsupported Warp graph modes: {', '.join(sorted(invalid_graph_modes))}")

    script = Path(__file__).resolve()
    cases = _build_cases(args)
    results = []
    for index, case in enumerate(cases, start=1):
        print(
            f"[{index}/{len(cases)}] {case.model} batch={case.batch_size} "
            f"impl={case.implementation} graph={case.graph_mode}",
            flush=True,
        )
        results.append(_run_subprocess_case(script, case))

    validation_results = []
    if "warp" in args.implementations and 512 in args.batch_sizes:
        for model in args.models:
            for graph_mode in args.warp_graph_modes:
                for capacity_scale in (1, 2):
                    validation_case = BenchmarkCase(
                        model=model,
                        batch_size=512,
                        implementation="warp",
                        graph_mode=graph_mode,
                        nsubsteps=args.nsubsteps,
                        seed=args.seed,
                        warmup_steps=0,
                        block_count=0,
                        steps_per_block=0,
                        latency_samples=10,
                        reset_samples=0,
                        warp_contact_capacity_per_env=args.warp_contact_capacity_per_env,
                        warp_constraint_capacity=args.warp_constraint_capacity,
                        capacity_scale=capacity_scale,
                        capacity_validation_only=True,
                    )
                    print(
                        f"[capacity x{capacity_scale}] {model} batch=512 graph={graph_mode}",
                        flush=True,
                    )
                    validation_results.append(_run_subprocess_case(script, validation_case))

    capacity_comparisons = _capacity_comparisons(validation_results)
    aggregate_workloads = _aggregate_workloads(results)
    capacity_gate_passed = bool(capacity_comparisons) and all(
        comparison["passed"] for comparison in capacity_comparisons
    )
    required_model_finiteness = _required_model_finiteness(
        results,
        required_models=tuple(args.models),
        batch_sizes=tuple(args.batch_sizes),
        implementations=tuple(args.implementations),
        graph_modes=tuple(args.warp_graph_modes),
    )
    payload = {
        "schema_version": 1,
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "command": " ".join(sys.argv),
            "host": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "jax_backend": jax.default_backend(),
            "jax_device": jax.devices()[0].device_kind,
        },
        "configuration": {
            "models": args.models,
            "batch_sizes": args.batch_sizes,
            "implementations": args.implementations,
            "warp_graph_modes": args.warp_graph_modes,
            "nsubsteps": args.nsubsteps,
            "seed": args.seed,
            "warmup_steps": args.warmup_steps,
            "block_count": args.block_count,
            "steps_per_block": args.steps_per_block,
            "latency_samples": args.latency_samples,
            "reset_samples": args.reset_samples,
            "environment_hot_path_fraction": ENVIRONMENT_HOT_PATH_FRACTION,
            "adoption_speedup_threshold": ADOPTION_SPEEDUP_THRESHOLD,
        },
        "results": results,
        "capacity_validation_results": validation_results,
        "analysis": {
            "capacity_comparisons": capacity_comparisons,
            "capacity_gate_passed": capacity_gate_passed,
            "required_model_finiteness": required_model_finiteness,
            "representative_workloads": aggregate_workloads,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.report.write_text(render_report(payload))
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report}")

    failed = [result for result in results if result.get("status") != "ok"]
    if failed:
        raise SystemExit(f"{len(failed)} benchmark case(s) failed; see {args.output}.")


if __name__ == "__main__":
    main()
