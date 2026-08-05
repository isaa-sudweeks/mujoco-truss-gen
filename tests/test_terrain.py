from __future__ import annotations

from dataclasses import replace

import jax
import mujoco
import numpy as np
import pytest

from mujoco_truss_gen import (
    DomainRandomizationConfig,
    MjxNodeVelocityEnv,
    MujocoNodeVelocityCommandEnv,
    TerrainConfig,
    TrussEnvConfig,
    add_terrain,
    generate_terrain,
    get_mujoco_spec,
    sample_model_terrain_height,
    view_terrain,
    view_terrain_explorer,
)

TERRAIN_KINDS = ("flat", "slope", "stairs", "waves", "rough")


def _small_config(kind: str = "rough", **overrides) -> TerrainConfig:
    values = dict(
        kind=kind,
        half_size=(4.0, 3.0),
        resolution=(25, 33),
        spawn_flat_radius=1.2,
        spawn_blend_width=0.5,
    )
    values.update(overrides)
    return TerrainConfig(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("kind", "unknown"),
        ("half_size", (0.0, 1.0)),
        ("resolution", (2, 3)),
        ("amplitude", -0.1),
        ("feature_size", 0.0),
        ("max_slope_degrees", 90.0),
        ("slope_angle_degrees", 90.0),
        ("stair_height", 0.0),
        ("stair_run", 0.0),
        ("spawn_flat_radius", -1.0),
        ("spawn_blend_width", -1.0),
        ("seed", -1),
        ("base_depth", 0.0),
        ("friction", (1.0, -1.0, 0.0)),
    ),
)
def test_terrain_config_rejects_invalid_values(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        TerrainConfig(**{field: value})


def test_terrain_config_rejects_spawn_region_outside_extent() -> None:
    with pytest.raises(ValueError, match="fit inside"):
        TerrainConfig(half_size=(2.0, 2.0), spawn_flat_radius=1.5, spawn_blend_width=0.5)


@pytest.mark.parametrize("kind", TERRAIN_KINDS)
def test_all_terrain_families_share_shape_extent_and_flat_spawn(kind: str) -> None:
    config = _small_config(kind)
    terrain = generate_terrain(config)

    assert terrain.heights.shape == config.resolution
    assert terrain.normalized_heights.shape == config.resolution
    np.testing.assert_allclose((terrain.x[0], terrain.x[-1]), (-4.0, 4.0))
    np.testing.assert_allclose((terrain.y[0], terrain.y[-1]), (-3.0, 3.0))
    assert np.all(np.isfinite(terrain.heights))
    assert np.all((terrain.normalized_heights >= 0.0) & (terrain.normalized_heights <= 1.0))

    xx, yy = np.meshgrid(terrain.x, terrain.y)
    inside_spawn = np.hypot(xx, yy) <= config.spawn_flat_radius
    np.testing.assert_array_equal(terrain.heights[inside_spawn], 0.0)


def test_flat_terrain_is_exactly_zero_for_every_seed() -> None:
    first = generate_terrain(_small_config("flat", seed=1))
    second = generate_terrain(_small_config("flat", seed=99))

    np.testing.assert_array_equal(first.heights, 0.0)
    np.testing.assert_array_equal(second.heights, first.heights)
    np.testing.assert_array_equal(first.normalized_heights, 0.0)


def test_rough_terrain_is_seeded_and_respects_slope_limit() -> None:
    config = _small_config("rough", seed=11, max_slope_degrees=20.0)
    first = generate_terrain(config)
    repeated = generate_terrain(config)
    different = generate_terrain(replace(config, seed=12))

    np.testing.assert_array_equal(first.heights, repeated.heights)
    assert not np.array_equal(first.heights, different.heights)
    assert np.max(np.abs(first.heights)) <= config.amplitude + 1e-12
    assert first.max_generated_slope <= np.tan(np.deg2rad(config.max_slope_degrees)) + 1e-12


def test_waves_slope_limit_uses_piecewise_planar_triangle_facets() -> None:
    config = TerrainConfig(
        kind="waves",
        half_size=(12.0, 12.0),
        resolution=(65, 65),
        amplitude=0.5,
        feature_size=0.8,
        max_slope_degrees=20.0,
    )
    terrain = generate_terrain(config)
    spacing_x, spacing_y = terrain.grid_spacing
    lower_left = terrain.heights[:-1, :-1]
    lower_right = terrain.heights[:-1, 1:]
    upper_left = terrain.heights[1:, :-1]
    upper_right = terrain.heights[1:, 1:]
    facet_slopes = np.concatenate(
        (
            np.hypot(
                (lower_right - lower_left) / spacing_x,
                (upper_right - lower_right) / spacing_y,
            ).ravel(),
            np.hypot(
                (upper_right - upper_left) / spacing_x,
                (upper_left - lower_left) / spacing_y,
            ).ravel(),
        )
    )

    maximum = np.tan(np.deg2rad(config.max_slope_degrees))
    assert np.max(facet_slopes) <= maximum + 1e-12
    assert terrain.max_generated_slope == pytest.approx(np.max(facet_slopes))


def test_default_world_remains_plane_and_explicit_terrain_is_heightfield() -> None:
    default_model = get_mujoco_spec("tetrahedron", realistic=False).compile()
    default_ground = default_model.geom("ground")
    assert int(default_ground.type[0]) == int(mujoco.mjtGeom.mjGEOM_PLANE)
    assert default_model.nhfield == 0

    config = _small_config("rough")
    terrain_model = get_mujoco_spec(
        "tetrahedron",
        realistic=False,
        terrain=config,
    ).compile()
    terrain_ground = terrain_model.geom("ground")
    assert int(terrain_ground.type[0]) == int(mujoco.mjtGeom.mjGEOM_HFIELD)
    assert terrain_model.nhfield == 1
    assert int(terrain_model.hfield_nrow[0]) == config.resolution[0]
    assert int(terrain_model.hfield_ncol[0]) == config.resolution[1]
    assert terrain_model.hfield_data.shape == (config.resolution[0] * config.resolution[1],)
    assert int(terrain_ground.contype[0]) == 1
    assert int(terrain_ground.conaffinity[0]) == 1


def test_add_terrain_replaces_existing_named_heightfield() -> None:
    spec = get_mujoco_spec("tetrahedron", realistic=False)
    add_terrain(spec, _small_config("rough", seed=3))
    replacement = add_terrain(spec, _small_config("waves", resolution=(17, 21)))
    model = spec.compile()

    assert model.nhfield == 1
    assert int(model.hfield_nrow[0]) == replacement.config.resolution[0]
    assert int(model.hfield_ncol[0]) == replacement.config.resolution[1]
    xy = np.array(((-2.5, -1.5), (0.0, 0.0), (2.0, 1.0)))
    np.testing.assert_allclose(
        sample_model_terrain_height(model, xy),
        replacement.height_at(xy[:, 0], xy[:, 1]),
        atol=1e-6,
    )


@pytest.mark.parametrize("kind", TERRAIN_KINDS)
def test_compiled_heightfield_round_trip_matches_generated_surface(kind: str) -> None:
    config = _small_config(kind)
    terrain = generate_terrain(config)
    model = get_mujoco_spec("tetrahedron", realistic=False, terrain=config).compile()
    xy = np.array(((-3.1, -2.2), (-0.4, 0.9), (0.0, 0.0), (2.7, 1.8)))

    np.testing.assert_allclose(
        sample_model_terrain_height(model, xy),
        terrain.height_at(xy[:, 0], xy[:, 1]),
        atol=2e-7,
    )


def test_native_reset_lifts_translated_robot_above_heightfield() -> None:
    config = TerrainConfig(
        kind="slope",
        half_size=(5.0, 5.0),
        resolution=(33, 33),
        slope_angle_degrees=15.0,
        spawn_flat_radius=1.3,
        spawn_blend_width=0.5,
    )
    env = MujocoNodeVelocityCommandEnv(
        TrussEnvConfig(
            get_mujoco_spec("tetrahedron", realistic=False, terrain=config),
            max_steps=2,
            domain_randomization=DomainRandomizationConfig(
                initial_translation_x_range=(2.5, 2.5),
            ),
        )
    )
    try:
        obs, _info = env.reset(seed=3)
        positions = env.mj_model.get_node_position_matrix()
        terrain_heights = sample_model_terrain_height(env.mj_model.model, positions[:, :2])
        clearances = positions[:, 2] - terrain_heights
        assert np.all(np.isfinite(obs))
        assert np.min(clearances - env.mj_model.initial_node_terrain_clearances) >= -1e-8

        obs, reward, terminated, truncated, _info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)
        assert not terminated
        assert not truncated
    finally:
        env.close()


def test_mjx_heightfield_reset_and_step_are_finite() -> None:
    config = TerrainConfig(
        kind="rough",
        half_size=(4.0, 4.0),
        resolution=(17, 17),
        spawn_flat_radius=1.3,
        spawn_blend_width=0.5,
        seed=5,
    )
    env = MjxNodeVelocityEnv(
        TrussEnvConfig(
            get_mujoco_spec("tetrahedron", realistic=False, terrain=config),
            max_steps=2,
        )
    )
    keys = jax.random.split(jax.random.key(4), 2)
    obs, state = jax.jit(env.reset)(keys)
    assert np.all(np.isfinite(np.asarray(obs)))

    obs, _state, reward, done, _info = jax.jit(env.step)(
        keys,
        state,
        np.zeros((2, env.action_size), dtype=np.float32),
    )
    assert np.all(np.isfinite(np.asarray(obs)))
    assert np.all(np.isfinite(np.asarray(reward)))
    assert not np.any(np.asarray(done))


def test_mjx_reset_ignores_nodes_outside_heightfield_extent() -> None:
    config = TerrainConfig(
        kind="rough",
        half_size=(3.0, 3.0),
        resolution=(17, 17),
        spawn_flat_radius=0.0,
        spawn_blend_width=0.0,
    )
    env = MjxNodeVelocityEnv(
        TrussEnvConfig(
            get_mujoco_spec("octahedron", scale=10.0, realistic=False, terrain=config),
            max_steps=2,
        )
    )
    keys = jax.random.split(jax.random.key(9), 1)
    obs, state = jax.jit(env.reset)(keys)

    assert np.all(np.isfinite(np.asarray(obs)))
    assert np.all(np.isfinite(np.asarray(state.data.qpos)))


def test_view_terrain_and_explorer_render_without_showing() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    config = _small_config("rough", resolution=(33, 33))
    fig, axes, terrain = view_terrain(config, show=False)
    explorer = view_terrain_explorer(config, show=False)
    try:
        assert len(axes) == 2
        assert "height map" in axes[0].get_title().lower()
        assert "centerline" in axes[1].get_title().lower()
        assert terrain.config == config

        explorer.widgets["kind"].set_active(2)
        assert explorer.terrain.config.kind == "stairs"
        assert explorer.map_axes.get_title() == "Stairs terrain height map"
        assert explorer.widgets["stair_height"].ax.get_visible()
        assert not explorer.widgets["amplitude"].ax.get_visible()
    finally:
        plt.close(fig)
        plt.close(explorer.figure)
