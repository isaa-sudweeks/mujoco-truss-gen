from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np

from mujoco_truss_gen.mujoco_model.constants import NODE_RADIUS
from mujoco_truss_gen.mujoco_model.terrain import (
    TerrainConfig,
    TerrainData,
    TerrainKind,
    generate_terrain,
)


def view_terrain(
    terrain: TerrainConfig | TerrainData | None = None,
    *,
    show: bool = True,
    figsize: tuple[float, float] = (12.0, 5.5),
):
    """Plot a terrain height map and +X centerline profile with Matplotlib.

    Returns ``(fig, axes, terrain_data)`` so callers can save or customize the
    figure. ``axes`` is ``(height_map_axes, centerline_axes)``.
    """

    pyplot, _widgets, circle = _matplotlib()
    terrain_data = terrain if isinstance(terrain, TerrainData) else generate_terrain(terrain)
    fig, axes = pyplot.subplots(1, 2, figsize=figsize)
    image = _draw_terrain(terrain_data, axes=axes, circle=circle)
    fig.colorbar(image, ax=axes[0], label="Terrain height (world units)", shrink=0.82)
    fig.tight_layout()
    if show:
        pyplot.show()
    return fig, tuple(axes), terrain_data


class TerrainExplorer:
    """Interactive Matplotlib controller returned by :func:`view_terrain_explorer`."""

    def __init__(
        self,
        config: TerrainConfig | None = None,
        *,
        figsize: tuple[float, float] = (14.0, 8.0),
    ) -> None:
        pyplot, widgets, circle = _matplotlib()
        self._pyplot = pyplot
        self._widgets_module = widgets
        self._circle = circle
        self.config = config or TerrainConfig(kind="rough", resolution=(65, 65))
        self.terrain = generate_terrain(self.config)
        self.figure = pyplot.figure(figsize=figsize)
        self.map_axes = self.figure.add_axes((0.06, 0.39, 0.47, 0.55))
        self.profile_axes = self.figure.add_axes((0.61, 0.39, 0.35, 0.55))
        self.axes = (self.map_axes, self.profile_axes)
        self._colorbar: Any | None = None
        self.widgets: dict[str, Any] = {}
        self._create_widgets()
        self._connect_widgets()
        self._update_widget_visibility()
        self._redraw()

    def _create_widgets(self) -> None:
        widgets = self._widgets_module
        self.widgets["kind"] = widgets.RadioButtons(
            self.figure.add_axes((0.02, 0.06, 0.09, 0.24)),
            ("flat", "slope", "stairs", "waves", "rough"),
            active=("flat", "slope", "stairs", "waves", "rough").index(self.config.kind),
        )
        resolution = int(self.config.resolution[0])
        resolution_options = ("33", "65", "129")
        active_resolution = (
            resolution_options.index(str(resolution))
            if str(resolution) in resolution_options
            else 1
        )
        self.widgets["resolution"] = widgets.RadioButtons(
            self.figure.add_axes((0.12, 0.06, 0.07, 0.24)),
            resolution_options,
            active=active_resolution,
        )
        slider_specs = {
            "amplitude": ("Amplitude", 0.01, 0.5, self.config.amplitude, 0.01),
            "feature_size": ("Feature size", 0.2, 3.0, self.config.feature_size, 0.05),
            "max_slope_degrees": (
                "Max slope (deg)",
                5.0,
                65.0,
                self.config.max_slope_degrees,
                1.0,
            ),
            "slope_angle_degrees": (
                "Slope angle (deg)",
                -25.0,
                25.0,
                self.config.slope_angle_degrees,
                1.0,
            ),
            "slope_direction_degrees": (
                "Direction (deg)",
                -180.0,
                180.0,
                self.config.slope_direction_degrees,
                5.0,
            ),
            "stair_height": ("Stair height", 0.02, 0.3, self.config.stair_height, 0.01),
            "stair_run": ("Stair run", 0.2, 2.0, self.config.stair_run, 0.05),
            "seed": ("Seed", 0.0, 99.0, float(self.config.seed), 1.0),
        }
        positions = (
            (0.27, 0.25, 0.27, 0.03),
            (0.66, 0.25, 0.27, 0.03),
            (0.27, 0.18, 0.27, 0.03),
            (0.66, 0.18, 0.27, 0.03),
            (0.27, 0.11, 0.27, 0.03),
            (0.66, 0.11, 0.27, 0.03),
            (0.27, 0.04, 0.27, 0.03),
            (0.66, 0.04, 0.27, 0.03),
        )
        for (name, (label, minimum, maximum, initial, step)), position in zip(
            slider_specs.items(), positions, strict=True
        ):
            self.widgets[name] = widgets.Slider(
                self.figure.add_axes(position),
                label,
                minimum,
                maximum,
                valinit=initial,
                valstep=step,
            )

    def _connect_widgets(self) -> None:
        self.widgets["kind"].on_clicked(self._on_kind_changed)
        self.widgets["resolution"].on_clicked(lambda _value: self._update())
        for name, widget in self.widgets.items():
            if name not in ("kind", "resolution"):
                widget.on_changed(lambda _value: self._update())

    def _on_kind_changed(self, _value: str) -> None:
        self._update_widget_visibility()
        self._update()

    def _update_widget_visibility(self) -> None:
        kind = str(self.widgets["kind"].value_selected)
        visible_for = {
            "amplitude": {"waves", "rough"},
            "feature_size": {"waves", "rough"},
            "max_slope_degrees": {"waves", "rough"},
            "slope_angle_degrees": {"slope"},
            "slope_direction_degrees": {"slope", "stairs"},
            "stair_height": {"stairs"},
            "stair_run": {"stairs"},
            "seed": {"rough"},
        }
        for name, kinds in visible_for.items():
            self.widgets[name].ax.set_visible(kind in kinds)

    def _update(self) -> None:
        resolution = int(self.widgets["resolution"].value_selected)
        self.config = replace(
            self.config,
            kind=cast(TerrainKind, str(self.widgets["kind"].value_selected)),
            resolution=(resolution, resolution),
            amplitude=float(self.widgets["amplitude"].val),
            feature_size=float(self.widgets["feature_size"].val),
            max_slope_degrees=float(self.widgets["max_slope_degrees"].val),
            slope_angle_degrees=float(self.widgets["slope_angle_degrees"].val),
            slope_direction_degrees=float(self.widgets["slope_direction_degrees"].val),
            stair_height=float(self.widgets["stair_height"].val),
            stair_run=float(self.widgets["stair_run"].val),
            seed=int(self.widgets["seed"].val),
        )
        self.terrain = generate_terrain(self.config)
        self._redraw()

    def _redraw(self) -> None:
        if self._colorbar is not None:
            self._colorbar.remove()
            self._colorbar = None
        for axes in self.axes:
            axes.clear()
        image = _draw_terrain(self.terrain, axes=self.axes, circle=self._circle)
        self._colorbar = self.figure.colorbar(
            image,
            ax=self.map_axes,
            label="Terrain height (world units)",
            shrink=0.76,
        )
        self.figure.canvas.draw_idle()

    def show(self) -> None:
        self._pyplot.show()


def view_terrain_explorer(
    config: TerrainConfig | None = None,
    *,
    show: bool = True,
    figsize: tuple[float, float] = (14.0, 8.0),
) -> TerrainExplorer:
    """Open an interactive terrain-family and roughness explorer."""

    explorer = TerrainExplorer(config, figsize=figsize)
    if show:
        explorer.show()
    return explorer


def _draw_terrain(
    terrain: TerrainData,
    *,
    axes: tuple[Any, Any] | np.ndarray,
    circle: Any,
) -> Any:
    map_axes, profile_axes = axes
    extent = (terrain.x[0], terrain.x[-1], terrain.y[0], terrain.y[-1])
    magnitude = max(
        abs(terrain.elevation_min), abs(terrain.elevation_min + terrain.elevation_range)
    )
    magnitude = max(magnitude, 1e-6)
    image = map_axes.imshow(
        terrain.heights,
        origin="lower",
        extent=extent,
        cmap="terrain",
        vmin=-magnitude,
        vmax=magnitude,
        aspect="equal",
    )
    if terrain.elevation_range > 1e-12:
        map_axes.contour(
            terrain.x,
            terrain.y,
            terrain.heights,
            levels=7,
            colors="black",
            linewidths=0.45,
            alpha=0.45,
        )
    config = terrain.config
    map_axes.add_patch(
        circle(
            (0.0, 0.0),
            config.spawn_flat_radius + config.spawn_blend_width,
            fill=False,
            linestyle="--",
            linewidth=1.4,
            edgecolor="tab:orange",
            label="spawn blend",
        )
    )
    map_axes.add_patch(
        circle(
            (0.0, 0.0),
            config.spawn_flat_radius,
            fill=False,
            linewidth=1.8,
            edgecolor="tab:red",
            label="flat spawn patch",
        )
    )
    map_axes.add_patch(
        circle(
            (0.0, 0.0),
            min(1.0, config.spawn_flat_radius * 0.7),
            color="tab:blue",
            alpha=0.18,
            label="illustrative robot footprint",
        )
    )
    map_axes.annotate(
        "+X",
        xy=(config.spawn_flat_radius + 1.5, 0.0),
        xytext=(0.3, 0.0),
        arrowprops={"arrowstyle": "->"},
        va="center",
    )
    map_axes.axhline(0.0, color="white", linewidth=0.8, linestyle=":", alpha=0.8)
    map_axes.set_title(f"{config.kind.capitalize()} terrain height map")
    map_axes.set_xlabel("X position (world units)")
    map_axes.set_ylabel("Y position (world units)")
    map_axes.legend(loc="upper right", fontsize=8)
    map_axes.format_coord = lambda x, y: (
        f"x={x:.3f}, y={y:.3f}, height={float(terrain.height_at(x, y)):.4f}"
    )

    row = int(np.argmin(np.abs(terrain.y)))
    profile_axes.plot(terrain.x, terrain.heights[row], color="tab:blue", linewidth=2.0)
    profile_axes.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    profile_axes.axvspan(
        -config.spawn_flat_radius,
        config.spawn_flat_radius,
        color="tab:red",
        alpha=0.1,
        label="flat spawn patch",
    )
    profile_axes.set_title(
        "Centerline profile\n"
        f"height {terrain.heights.min():.3f} to {terrain.heights.max():.3f}; "
        f"max slope {np.rad2deg(np.arctan(terrain.max_generated_slope)):.1f}°"
    )
    profile_axes.set_xlabel("X position (world units)")
    profile_axes.set_ylabel("Terrain height (world units)")
    profile_axes.grid(alpha=0.2)
    profile_axes.legend(loc="best", fontsize=8)
    spacing_x, spacing_y = terrain.grid_spacing
    profile_axes.text(
        0.02,
        0.02,
        "\n".join(
            (
                f"grid: {config.resolution[1]} x {config.resolution[0]} samples",
                f"extent: {2.0 * config.half_size[0]:.2f} x "
                f"{2.0 * config.half_size[1]:.2f} world units",
                f"spacing: dx={spacing_x:.3f}, dy={spacing_y:.3f}",
                f"feature size / nominal node radius: {config.feature_size / NODE_RADIUS:.1f}",
                "Fixed dimensions across terrain families (MJX-compatible)",
            )
        ),
        transform=profile_axes.transAxes,
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.78},
    )
    profile_axes.format_coord = lambda x, _y: (
        f"x={x:.3f}, height={float(terrain.height_at(x, 0.0)):.4f}"
    )
    return image


def _matplotlib() -> tuple[Any, Any, Any]:
    try:
        import matplotlib.pyplot as pyplot
        import matplotlib.widgets as widgets
        from matplotlib.patches import Circle
    except ImportError as exc:
        raise ImportError(
            "Terrain visualization requires matplotlib. Install it with: "
            'pip install "mujoco-truss-gen[terrain]"'
        ) from exc
    return pyplot, widgets, Circle
