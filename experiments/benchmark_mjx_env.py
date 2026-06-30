from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

from mujoco_truss_gen import MjxNodeVelocityEnv, TrussEnvConfig, get_mujoco_spec


def benchmark(
    preset: str,
    *,
    realistic: bool,
    batch_sizes: tuple[int, ...],
    iterations: int,
    nsubsteps: int,
) -> None:
    env = MjxNodeVelocityEnv(
        TrussEnvConfig(
            get_mujoco_spec(preset, realistic=realistic),
            nsubsteps=nsubsteps,
        )
    )
    compiled_reset = jax.jit(env.reset)
    compiled_step = jax.jit(env.step)

    print(f"backend={jax.default_backend()} preset={preset} realistic={realistic}")
    for batch_size in batch_sizes:
        keys = jax.random.split(jax.random.key(batch_size), batch_size)
        actions = jnp.zeros((batch_size, env.action_size), dtype=jnp.float32)
        _, state = compiled_reset(keys)
        state.data.qpos.block_until_ready()

        _, state, _, _, _ = compiled_step(keys, state, actions)
        state.data.qpos.block_until_ready()

        start = time.perf_counter()
        for _ in range(iterations):
            _, state, _, _, _ = compiled_step(keys, state, actions)
        state.data.qpos.block_until_ready()
        elapsed = time.perf_counter() - start
        environment_steps = batch_size * iterations
        print(
            f"batch={batch_size:5d} elapsed={elapsed:8.3f}s "
            f"env_steps_per_second={environment_steps / elapsed:12.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark warm JIT MJX environment throughput.")
    parser.add_argument("--preset", default="tetrahedron")
    parser.add_argument("--abstract", action="store_true")
    parser.add_argument("--batch-sizes", default="1,16,64,256")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--nsubsteps", type=int, default=1)
    args = parser.parse_args()
    batch_sizes = tuple(int(value) for value in args.batch_sizes.split(",") if value)
    benchmark(
        args.preset,
        realistic=not args.abstract,
        batch_sizes=batch_sizes,
        iterations=args.iterations,
        nsubsteps=args.nsubsteps,
    )


if __name__ == "__main__":
    main()
