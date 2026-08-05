from mujoco_truss_gen import TerrainConfig, view_terrain_explorer


def main() -> None:
    view_terrain_explorer(
        TerrainConfig(
            kind="rough",
            resolution=(65, 65),
            amplitude=0.15,
            feature_size=0.8,
            seed=7,
        )
    )


if __name__ == "__main__":
    main()
