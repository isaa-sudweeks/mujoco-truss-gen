from __future__ import annotations

import numpy as np

from mujoco_truss_gen import MujocoTrussEnv, TrussEnvConfig, get_mujoco_spec, save_xml


def main() -> None:
    node_dict = {
        "node_1": [0.0, 0.0, 0.2],
        "node_2": [0.8, 0.0, 0.2],
        "node_3": [0.4, 0.7, 0.2],
    }
    triangle_dict = {
        "triangle_1": ["node_1", "node_2", "node_3", "node_1"],
    }

    spec = get_mujoco_spec(node_dict, triangle_dict, realistic=False)
    xml_path = save_xml(spec, "custom_truss.xml")
    print(f"Wrote {xml_path}")

    env = MujocoTrussEnv(TrussEnvConfig(spec, max_steps=100, nsubsteps=1, speed=0.01))
    try:
        obs, info = env.reset(seed=0)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"obs shape: {obs.shape}")
        print(f"reward: {reward:.4f}")
        print(f"terminated: {terminated}, truncated: {truncated}")
        print(f"critical_eig: {info['critical_eig']:.4f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
