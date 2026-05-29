from __future__ import annotations

import xml.etree.ElementTree as ET
from copy import deepcopy

import mujoco
import numpy as np
import pytest

from mujoco_truss_gen import (
    PRESETS,
    AccelerometerConfig,
    MujocoModel,
    MujocoNodeVelocityCommandEnv,
    MujocoRelativeObsEnv,
    MujocoTrussEnv,
    MujocoVelocityCommandEnv,
    NodeVelocityController,
    TrussEnvConfig,
    TrussPhysicalParameters,
    get_edge_index,
    get_icosahedron_definition,
    get_mujoco_spec,
    get_node_features,
    get_preset_definition,
    get_route_lengths,
    save_xml,
)
from mujoco_truss_gen.mujoco_model.constants import (
    ACTIVE_NODE_MASS,
    EDGE_TENDON_WIDTH,
    NODE_RADIUS,
    PASSIVE_NODE_MASS,
)
from mujoco_truss_gen.mujoco_model.io_viewer import (
    NodeVelocityViewerState,
    _apply_terminal_command,
)


def test_generated_spec_runs_in_all_builtin_envs() -> None:
    spec = get_mujoco_spec("octahedron", realistic=False)

    for env_cls in (MujocoTrussEnv, MujocoRelativeObsEnv, MujocoVelocityCommandEnv):
        env = env_cls(TrussEnvConfig(spec, max_steps=3, nsubsteps=1, speed=0.01))
        try:
            obs, _ = env.reset(seed=123)
            assert env.observation_space.contains(obs)

            action = np.zeros(env.action_space.shape, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)

            assert env.observation_space.contains(obs)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert "critical_eig" in info
        finally:
            env.close()


def test_builtin_presets_compile() -> None:
    assert {"octahedron", "icosahedron", "tetrahedron"} <= set(PRESETS)

    for preset_name in PRESETS:
        get_mujoco_spec(preset_name, realistic=False).compile()

    for preset_name in ("octahedron", "icosahedron", "solar_array"):
        get_mujoco_spec(preset_name, realistic=True).compile()

    get_mujoco_spec("tetrahedron", realistic=True).compile()


def test_builtin_preset_definitions_support_unit_scale() -> None:
    unscaled_nodes, unscaled_structure = get_preset_definition("octahedron")
    same_nodes, same_structure = get_preset_definition("octahedron", scale=1.0)
    scaled_nodes, scaled_structure = get_preset_definition("octahedron", scale=2.5)

    assert same_nodes == unscaled_nodes
    assert same_structure == unscaled_structure
    assert scaled_structure == unscaled_structure
    for node_name, position in unscaled_nodes.items():
        np.testing.assert_allclose(scaled_nodes[node_name], np.asarray(position) * 2.5)


def test_scaled_named_preset_compiles() -> None:
    get_mujoco_spec("tetrahedron", scale=0.5, realistic=False).compile()
    get_mujoco_spec("octahedron", scale=2.5, realistic=False).compile()
    get_mujoco_spec("octahedron", scale=2.0, realistic=True).compile()


def test_scaled_abstract_preset_keeps_control_values_unscaled() -> None:
    root = ET.fromstring(get_mujoco_spec("octahedron", scale=2.5, realistic=False).to_xml())

    tendon = root.find(".//tendon/spatial[@name='tendon_node_1_node_2']")
    assert tendon is not None
    np.testing.assert_allclose(_xml_vector(tendon.get("range", "")), [0.5, 5.0])

    actuator = root.find(".//actuator/general[@name='act_12']")
    assert actuator is not None
    np.testing.assert_allclose(_xml_vector(actuator.get("ctrlrange", "")), [-0.05, 0.05])
    np.testing.assert_allclose(_xml_vector(actuator.get("actrange", "")), [0.0, 3.0])


def test_preset_scale_must_be_positive() -> None:
    with pytest.raises(ValueError, match="scale must be greater than zero"):
        get_preset_definition("octahedron", scale=0.0)

    with pytest.raises(ValueError, match="scale is only supported"):
        get_mujoco_spec({"node_1": [0.0, 0.0, 0.1]}, {}, scale=2.0)


def test_generated_world_uses_professional_scene_defaults() -> None:
    spec = get_mujoco_spec("tetrahedron", realistic=False)
    root = ET.fromstring(spec.to_xml())

    option = root.find("./option")
    assert option is not None
    assert option.get("integrator") == "implicitfast"
    assert spec.compile().opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    ground = root.find("./worldbody/geom[@name='ground']")
    assert ground is not None
    assert ground.get("type") == "plane"
    assert ground.get("material") == "ground_grid"

    ground_texture = root.find("./asset/texture[@name='ground_checker']")
    assert ground_texture is not None
    assert ground_texture.get("builtin") == "checker"

    skybox = root.find("./asset/texture[@name='skybox']")
    assert skybox is not None
    assert skybox.get("type") == "skybox"

    light_names = {
        light.get("name") for light in root.findall("./worldbody/light")
    }
    assert {"key", "fill"} <= light_names


def test_generated_spec_uses_firehose_steel_and_black_materials() -> None:
    root = ET.fromstring(get_mujoco_spec("octahedron", realistic=True).to_xml())

    firehose_material = root.find("./asset/material[@name='blue_firehose']")
    assert firehose_material is not None
    assert firehose_material.get("texture") is None
    np.testing.assert_allclose(
        _xml_vector(firehose_material.get("rgba", "")),
        [0.0, 0.1804, 0.3647, 1.0],
    )
    assert float(firehose_material.get("reflectance", "nan")) == pytest.approx(0.01)
    assert float(firehose_material.get("specular", "nan")) == pytest.approx(0.08)

    for tendon in root.findall(".//tendon/spatial"):
        if tendon.get("name", "").startswith("Perimeter_Constraint_"):
            assert tendon.get("material") is None
            assert float(tendon.get("width", "nan")) == pytest.approx(0.0001)
            np.testing.assert_allclose(
                _xml_vector(tendon.get("rgba", "")),
                [0.0, 0.0, 0.0, 0.0],
            )
        else:
            assert tendon.get("material") == "blue_firehose"

    rod_geom = next(
        body.find("./geom")
        for body in root.findall(".//body")
        if body.get("name", "").startswith("rod_")
    )
    assert rod_geom is not None
    assert rod_geom.get("material") == "connector_steel"

    node_geom = root.find(".//body[@name='node_1']/geom")
    assert node_geom is not None
    assert node_geom.get("material") == "node_black"
    np.testing.assert_allclose(
        _xml_vector(node_geom.get("rgba", "")),
        [0.18, 0.18, 0.18, 1.0],
    )


def test_realistic_spec_adds_accelerometers_to_each_generated_node() -> None:
    root = ET.fromstring(get_mujoco_spec("octahedron", realistic=True).to_xml())

    node_site_names = {
        site.get("name")
        for site in root.findall(".//body/site")
        if site.get("name", "").startswith("node_")
    }
    accelerometers = root.findall("./sensor/accelerometer")

    assert {sensor.get("site") for sensor in accelerometers} == node_site_names
    assert {sensor.get("name") for sensor in accelerometers} == {
        f"accel_{node_name}" for node_name in node_site_names
    }


def test_realistic_accelerometer_config_is_passed_to_mujoco() -> None:
    spec = get_mujoco_spec(
        "octahedron",
        realistic=True,
        accelerometer_config=AccelerometerConfig(
            noise=0.03,
            cutoff=25.0,
            nsample=3,
            delay=0.01,
            name_prefix="imu_accel",
        ),
    )
    root = ET.fromstring(spec.to_xml())
    sensor = root.find("./sensor/accelerometer")

    assert sensor is not None
    assert sensor.get("name", "").startswith("imu_accel_node_")
    assert float(sensor.get("noise", "nan")) == pytest.approx(0.03)
    assert float(sensor.get("cutoff", "nan")) == pytest.approx(25.0)
    assert int(sensor.get("nsample", "0")) == 3
    assert float(sensor.get("delay", "nan")) == pytest.approx(0.01)
    spec.compile()


def test_realistic_accelerometers_can_be_disabled() -> None:
    root = ET.fromstring(
        get_mujoco_spec("octahedron", realistic=True, accelerometer_config=None).to_xml()
    )

    assert root.find("./sensor/accelerometer") is None


def test_icosahedron_definition_shape() -> None:
    node_dict, triangle_dict = get_icosahedron_definition()

    assert len(node_dict) == 12
    assert len(triangle_dict) == 20
    for triangle_nodes in triangle_dict.values():
        assert len(triangle_nodes) == 4
        assert triangle_nodes[3] in triangle_nodes[:3]


def test_env_accepts_xml_path(tmp_path) -> None:
    spec = get_mujoco_spec("octahedron", realistic=False)
    xml_path = tmp_path / "model.xml"
    xml_path.write_text(spec.to_xml(), encoding="utf-8")

    env = MujocoTrussEnv(xml_path, max_steps=1)
    try:
        env.reset(seed=7)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, truncated, _ = env.step(action)
        assert truncated
    finally:
        env.close()


def test_save_xml_relative_path_uses_current_working_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    spec = get_mujoco_spec("octahedron", realistic=False)

    xml_path = save_xml(spec, "generated/octahedron.xml")

    assert xml_path == tmp_path / "generated" / "octahedron.xml"
    assert xml_path.exists()
    assert xml_path.read_text(encoding="utf-8").lstrip().startswith("<mujoco")


def test_custom_dictionary_spec_compiles_and_runs() -> None:
    node_dict = {
        "node_1": [0.0, 0.0, 0.2],
        "node_2": [0.8, 0.0, 0.2],
        "node_3": [0.4, 0.7, 0.2],
    }
    triangle_dict = {
        "triangle_1": ["node_1", "node_2", "node_3", "node_1"],
    }

    spec = get_mujoco_spec(node_dict, triangle_dict, realistic=False)
    env = MujocoTrussEnv(TrussEnvConfig(spec, max_steps=2, nsubsteps=1, speed=0.01))
    try:
        obs, _ = env.reset(seed=11)
        assert env.observation_space.contains(obs)

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert not truncated
        assert "critical_eig" in info
    finally:
        env.close()


def test_triangle_node_masses_follow_active_and_passive_roles() -> None:
    node_dict = {
        "node_1": [0.0, 0.0, 0.2],
        "node_2": [0.8, 0.0, 0.2],
        "node_3": [0.4, 0.7, 0.2],
    }
    triangle_dict = {
        "triangle_1": ["node_1", "node_2", "node_3", "node_1"],
    }

    root = ET.fromstring(get_mujoco_spec(node_dict, triangle_dict, realistic=False).to_xml())

    passive_geom = root.find(".//body[@name='node_1']/geom")
    active_geom = root.find(".//body[@name='node_2']/geom")
    assert passive_geom is not None
    assert active_geom is not None
    assert float(passive_geom.get("mass", "nan")) == pytest.approx(PASSIVE_NODE_MASS)
    assert float(active_geom.get("mass", "nan")) == pytest.approx(ACTIVE_NODE_MASS)


def test_physical_parameters_override_generated_truss_values() -> None:
    params = TrussPhysicalParameters(
        node_radius=0.12,
        active_node_mass=0.33,
        passive_node_mass=0.44,
        abstract_actuator_kp=1234.0,
        actuator_dampratio=0.75,
        actuator_ctrl_range=[-0.2, 0.2],
        default_actuator_range=[0.1, 1.7],
        tendon_range_max_factor=3.0,
        edge_tendon_width=0.08,
        perimeter_constraint_tendon_width=0.002,
        tendon_constraint_solref=[0.03, 0.8],
        tendon_constraint_solimp=[0.85, 0.93, 0.002],
    )
    node_dict = {
        "node_1": [0.0, 0.0, 0.2],
        "node_2": [0.8, 0.0, 0.2],
        "node_3": [0.4, 0.7, 0.2],
    }
    triangle_dict = {
        "triangle_1": ["node_1", "node_2", "node_3", "node_1"],
    }

    root = ET.fromstring(
        get_mujoco_spec(
            node_dict,
            triangle_dict,
            realistic=False,
            physical_params=params,
        ).to_xml()
    )

    passive_geom = root.find(".//body[@name='node_1']/geom")
    active_geom = root.find(".//body[@name='node_2']/geom")
    edge_tendon = root.find(".//tendon/spatial[@name='tendon_node_1_node_2']")
    perimeter_tendon = root.find(".//tendon/spatial[@name='Perimeter_Constraint_0']")
    actuator = root.find(".//actuator/general")
    constraint = root.find(".//equality/tendon[@name='Perimeter_Constraint_0']")

    assert passive_geom is not None
    assert active_geom is not None
    assert edge_tendon is not None
    assert perimeter_tendon is not None
    assert actuator is not None
    assert constraint is not None

    assert float(passive_geom.get("mass", "nan")) == pytest.approx(0.44)
    assert float(active_geom.get("mass", "nan")) == pytest.approx(0.33)
    np.testing.assert_allclose(_xml_vector(active_geom.get("size", "")), [0.12])
    np.testing.assert_allclose(_xml_vector(edge_tendon.get("range", "")), [0.5, 2.4])
    assert float(edge_tendon.get("width", "nan")) == pytest.approx(0.08)
    assert float(perimeter_tendon.get("width", "nan")) == pytest.approx(0.002)
    np.testing.assert_allclose(_xml_vector(actuator.get("ctrlrange", "")), [-0.2, 0.2])
    np.testing.assert_allclose(_xml_vector(actuator.get("actrange", "")), [0.1, 1.7])
    np.testing.assert_allclose(_xml_vector(actuator.get("gainprm", ""))[:1], [1234.0])
    np.testing.assert_allclose(_xml_vector(actuator.get("biasprm", ""))[1:3], [-1234.0, 0.75])
    np.testing.assert_allclose(_xml_vector(constraint.get("solref", "")), [0.03, 0.8])
    np.testing.assert_allclose(
        _xml_vector(constraint.get("solimp", ""))[:3],
        [0.85, 0.93, 0.002],
    )


def test_custom_triangle_nodes_are_lifted_above_ground() -> None:
    node_dict = {
        "node_1": [0.0, 0.0, -0.8],
        "node_2": [0.8, 0.0, -0.4],
        "node_3": [0.4, 0.7, -0.6],
    }
    triangle_dict = {
        "triangle_1": ["node_1", "node_2", "node_3", "node_1"],
    }

    for realistic in (False, True):
        env = MujocoTrussEnv(
            TrussEnvConfig(
                get_mujoco_spec(node_dict, triangle_dict, realistic=realistic),
                max_steps=1,
                nsubsteps=1,
            )
        )
        try:
            for seed in range(5):
                env.reset(seed=seed)
                node_z = env.mj_model.get_node_position_matrix()[:, 2]
                assert float(np.min(node_z)) >= NODE_RADIUS
        finally:
            env.close()


def test_custom_routed_shape_nodes_are_lifted_above_ground() -> None:
    node_dict = {
        "node_1": [0.0, 0.0, -0.8],
        "node_2": [0.8, 0.0, -0.4],
        "node_3": [0.8, 0.8, -0.6],
        "node_4": [0.0, 0.8, -0.5],
    }
    shape_dict = {
        "quad_1": {
            "route": ["node_1", "node_2", "node_3", "node_4", "node_1"],
            "active_edges": [["node_1", "node_2"], ["node_4", "node_1"]],
        },
    }

    env = MujocoTrussEnv(
        TrussEnvConfig(
            get_mujoco_spec(node_dict, shape_dict, realistic=False),
            max_steps=1,
            nsubsteps=1,
        )
    )
    try:
        for seed in range(5):
            env.reset(seed=seed)
            node_z = env.mj_model.get_node_position_matrix()[:, 2]
            assert float(np.min(node_z)) >= NODE_RADIUS
    finally:
        env.close()


def test_builtin_presets_reset_with_nodes_above_ground() -> None:
    for preset_name in PRESETS:
        realistic_values = (False, True) if preset_name != "tetrahedron" else (False,)
        for realistic in realistic_values:
            env = MujocoTrussEnv(
                TrussEnvConfig(
                    get_mujoco_spec(preset_name, realistic=realistic),
                    max_steps=1,
                    nsubsteps=1,
                )
            )
            try:
                for seed in range(10):
                    env.reset(seed=seed)
                    node_z = env.mj_model.get_node_position_matrix()[:, 2]
                    assert float(np.min(node_z)) >= NODE_RADIUS
            finally:
                env.close()


def test_realistic_octahedron_starts_above_collapse_threshold() -> None:
    env = MujocoTrussEnv(
        TrussEnvConfig(
            get_mujoco_spec("octahedron", realistic=True),
            max_steps=1,
            nsubsteps=1,
        )
    )
    try:
        env.reset(seed=0)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, terminated, _, info = env.step(action)

        assert info["critical_eig"] > env.config.critical_eig_threshold
        assert not terminated
    finally:
        env.close()


def test_octahedron_stays_finite_and_above_ground_under_zero_action() -> None:
    env = MujocoTrussEnv(
        TrussEnvConfig(
            get_mujoco_spec("octahedron", realistic=False),
            max_steps=250,
            nsubsteps=1,
            speed=0.01,
        )
    )
    try:
        obs, _ = env.reset(seed=23)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        ground_contact_tolerance = 0.015

        for _ in range(200):
            obs, reward, terminated, truncated, info = env.step(action)
            node_z = env.mj_model.get_node_position_matrix()[:, 2]

            assert np.all(np.isfinite(obs))
            assert np.isfinite(reward)
            assert np.isfinite(info["critical_eig"])
            assert float(np.min(node_z)) >= NODE_RADIUS - ground_contact_tolerance
            assert not terminated
            assert not truncated
    finally:
        env.close()


def test_custom_triangle_validation_reports_unknown_nodes() -> None:
    node_dict = {
        "node_1": [0.0, 0.0, 0.2],
        "node_2": [0.8, 0.0, 0.2],
        "node_3": [0.4, 0.7, 0.2],
    }
    triangle_dict = {
        "triangle_1": ["node_1", "node_2", "node_missing", "node_1"],
    }

    with pytest.raises(ValueError, match="Triangle 'triangle_1' references unknown node"):
        get_mujoco_spec(node_dict, triangle_dict)


def test_custom_triangle_validation_reports_bad_passive_node() -> None:
    node_dict = {
        "node_1": [0.0, 0.0, 0.2],
        "node_2": [0.8, 0.0, 0.2],
        "node_3": [0.4, 0.7, 0.2],
        "node_4": [0.4, -0.7, 0.2],
    }
    triangle_dict = {
        "triangle_1": ["node_1", "node_2", "node_3", "node_4"],
    }

    with pytest.raises(ValueError, match="passive node 'node_4' must be one"):
        get_mujoco_spec(node_dict, triangle_dict)


def test_custom_node_validation_reports_bad_position() -> None:
    node_dict = {
        "node_1": [0.0, 0.0],
        "node_2": [0.8, 0.0, 0.2],
        "node_3": [0.4, 0.7, 0.2],
    }
    triangle_dict = {
        "triangle_1": ["node_1", "node_2", "node_3", "node_1"],
    }

    with pytest.raises(ValueError, match="Node 'node_1' position"):
        get_mujoco_spec(node_dict, triangle_dict)


def test_generation_does_not_mutate_custom_dictionaries() -> None:
    node_dict = {
        "node_1": [0.0, 0.0, 0.2],
        "node_2": [0.8, 0.0, 0.2],
        "node_3": [0.4, 0.7, 0.2],
        "node_4": [0.4, -0.7, 0.2],
    }
    triangle_dict = {
        "triangle_1": ["node_1", "node_2", "node_3", "node_1"],
        "triangle_2": ["node_1", "node_4", "node_2", "node_1"],
    }
    original_nodes = deepcopy(node_dict)
    original_triangles = deepcopy(triangle_dict)

    get_mujoco_spec(node_dict, triangle_dict, realistic=False).compile()
    assert node_dict == original_nodes
    assert triangle_dict == original_triangles

    get_mujoco_spec(node_dict, triangle_dict, realistic=True).compile()
    assert node_dict == original_nodes
    assert triangle_dict == original_triangles


def test_realistic_angle_bisector_controller_aligns_connector_rods() -> None:
    node_dict = {
        "node_1": [0.0, 0.0, 0.2],
        "node_2": [0.8, 0.0, 0.2],
        "node_3": [0.4, 0.7, 0.2],
        "node_4": [0.4, -0.7, 0.2],
    }
    triangle_dict = {
        "triangle_1": ["node_1", "node_2", "node_3", "node_1"],
        "triangle_2": ["node_1", "node_4", "node_2", "node_1"],
    }

    env = MujocoTrussEnv(
        TrussEnvConfig(
            get_mujoco_spec(node_dict, triangle_dict, realistic=True),
            max_steps=2,
            nsubsteps=1,
            speed=0.01,
        )
    )
    try:
        env.reset(seed=17)
        controller = env.mj_model.angle_bisector_controller

        assert controller.enabled
        assert {
            target.node_name for target in controller.targets
        } == {"node_1", "node_2", "node_1_tri_triangle_2", "node_2_tri_triangle_2"}
        assert env.action_space.shape == (4,)
        assert env.mj_model.model.nu == 8
        assert all(
            name.startswith("bisector_act_")
            for name in env.mj_model.internal_actuator_names
        )
        assert not any(
            name.startswith("bisector_act_")
            for name in env.mj_model.external_actuator_names
        )

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        env.step(action)

        for target in controller.targets:
            node_pos = env.mj_model.data.site_xpos[target.node_site_id]
            neighbor_a = env.mj_model.data.site_xpos[target.neighbor_site_ids[0]]
            neighbor_b = env.mj_model.data.site_xpos[target.neighbor_site_ids[1]]
            dir_a = _unit(neighbor_a - node_pos)
            dir_b = _unit(neighbor_b - node_pos)
            bisector = _unit(dir_a + dir_b)

            tip_site_id = mujoco.mj_name2id(
                env.mj_model.model,
                mujoco.mjtObj.mjOBJ_SITE,
                f"tip_site_{target.node_name}",
            )
            rod_direction = _unit(env.mj_model.data.site_xpos[tip_site_id] - node_pos)

            assert float(np.dot(rod_direction, bisector)) == pytest.approx(-1.0, abs=1e-4)
    finally:
        env.close()


def test_realistic_node_box_face_normal_points_to_connector_ball() -> None:
    node_dict = {
        "node_1": [0.0, 0.0, 0.2],
        "node_2": [0.8, 0.0, 0.2],
        "node_3": [0.4, 0.7, 0.2],
        "node_4": [0.4, -0.7, 0.2],
    }
    triangle_dict = {
        "triangle_1": ["node_1", "node_2", "node_3", "node_1"],
        "triangle_2": ["node_1", "node_4", "node_2", "node_1"],
    }

    root = ET.fromstring(get_mujoco_spec(node_dict, triangle_dict, realistic=True).to_xml())

    for node_name in ("node_1", "node_2", "node_1_tri_triangle_2", "node_2_tri_triangle_2"):
        node_body = root.find(f".//body[@name='{node_name}']")
        assert node_body is not None

        box_geom = node_body.find("./geom[@type='box']")
        assert box_geom is not None
        face_normal = _quat_rotate_x(_xml_vector(box_geom.get("quat", "1 0 0 0")))

        tip_site = node_body.find(f"./body[@name='rod_{node_name}']/site")
        assert tip_site is not None
        connector_direction = _unit(_xml_vector(tip_site.get("pos", "")))

        assert float(np.dot(face_normal, connector_direction)) == pytest.approx(1.0)


def test_routed_shape_spec_compiles_and_runs() -> None:
    node_dict = {
        "node_1": [0.0, 0.0, 0.2],
        "node_2": [0.8, 0.0, 0.2],
        "node_3": [0.8, 0.8, 0.2],
        "node_4": [0.0, 0.8, 0.2],
    }
    shape_dict = {
        "quad_1": {
            "route": ["node_1", "node_2", "node_3", "node_4", "node_1"],
            "active_edges": [["node_1", "node_2"], ["node_4", "node_1"]],
        },
    }

    assert get_route_lengths(node_dict, shape_dict) == {"quad_1": 3.2}
    spec = get_mujoco_spec(node_dict, shape_dict, realistic=False)
    assert get_edge_index(spec).shape == (2, 8)
    root = ET.fromstring(spec.to_xml())
    assert root.find(".//equality/tendon[@name='Route_Length_Constraint_quad_1']") is None
    actuator_names = {
        actuator.get("name")
        for actuator in root.findall(".//actuator/general")
    }
    assert actuator_names == {"act_12", "act_23", "act_34", "act_14"}

    env = MujocoTrussEnv(TrussEnvConfig(spec, max_steps=2, nsubsteps=1, speed=0.01))
    try:
        obs, _ = env.reset(seed=13)
        assert env.observation_space.contains(obs)
        assert env.action_space.shape == (4,)

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, _, _, _, info = env.step(action)

        assert env.observation_space.contains(obs)
        assert "critical_eig" in info
    finally:
        env.close()


def test_realistic_routed_shape_clones_nodes_and_adds_bisector_controller() -> None:
    node_dict, shape_dict = get_preset_definition("tetrahedron")
    spec = get_mujoco_spec(node_dict, shape_dict, realistic=True)
    root = ET.fromstring(spec.to_xml())

    route_sites = {
        spatial.get("name"): [site.get("site") for site in spatial.findall("site")]
        for spatial in root.findall("./tendon/spatial")
        if spatial.get("name", "").startswith("route_")
    }
    assert route_sites == {
        "route_path_1": ["node_1", "node_2", "node_4", "node_3"],
        "route_path_2": [
            "node_2_route_path_2_0",
            "node_3_route_path_2_1",
            "node_1_route_path_2_2",
            "node_4_route_path_2_3",
        ],
    }
    assert {
        body.get("name")
        for body in root.findall("./worldbody/body")
        if body.get("name", "").startswith("connector_ball_")
    } == {
        "connector_ball_node_1",
        "connector_ball_node_2",
        "connector_ball_node_3",
        "connector_ball_node_4",
    }

    model = MujocoModel(spec)
    controller = model.angle_bisector_controller
    assert controller.enabled
    assert {target.node_name for target in controller.targets} == {
        "node_2",
        "node_4",
        "node_3_route_path_2_1",
        "node_1_route_path_2_2",
    }
    assert len(model.external_actuator_names) == 6
    assert all(
        not name.startswith("bisector_act_") for name in model.external_actuator_names
    )
    for tendon_name, edge_length in model.get_edge_length_dict().items():
        if tendon_name.startswith("tendon_"):
            assert edge_length == pytest.approx(1.0, abs=0.05)


def test_realistic_routed_passive_cylinders_face_connector_rods() -> None:
    root = ET.fromstring(get_mujoco_spec("tetrahedron", realistic=True).to_xml())
    passive_nodes = set()
    for spatial in root.findall("./tendon/spatial"):
        if not spatial.get("name", "").startswith("route_"):
            continue
        sites = spatial.findall("site")
        passive_nodes.update((sites[0].get("site"), sites[-1].get("site")))
    routed_nodes = [
        body
        for body in root.findall("./worldbody/body")
        if body.get("name", "").startswith("node_")
        and body.find(f"./body[@name='rod_{body.get('name')}']") is not None
    ]
    assert routed_nodes

    for node_body in routed_nodes:
        node_name = node_body.get("name")
        assert node_name is not None

        hinge_joint = node_body.find(f"./joint[@name='{node_name}_z_hinge']")
        assert hinge_joint is not None

        rod_body = node_body.find(f"./body[@name='rod_{node_name}']")
        assert rod_body is not None
        assert rod_body.find("./joint") is None

        node_geom = node_body.find("./geom")
        assert node_geom is not None
        if node_name in passive_nodes:
            assert node_geom.get("type") == "cylinder"
            size = _xml_vector(node_geom.get("size", ""))
            assert float(size[0]) == pytest.approx(EDGE_TENDON_WIDTH)
            face_normal = _quat_rotate_z(_xml_vector(node_geom.get("quat", "1 0 0 0")))
        else:
            assert node_geom.get("type") == "box"
            face_normal = _quat_rotate_x(_xml_vector(node_geom.get("quat", "1 0 0 0")))

        tip_site = rod_body.find(f"./site[@name='tip_site_{node_name}']")
        assert tip_site is not None
        connector_direction = _unit(_xml_vector(tip_site.get("pos", "")))

        assert float(np.dot(face_normal, connector_direction)) == pytest.approx(
            1.0,
            abs=1e-5,
        )


def test_realistic_routed_connector_rods_start_on_angle_bisectors() -> None:
    node_dict, shape_dict = get_preset_definition("tetrahedron")
    spec = get_mujoco_spec(node_dict, shape_dict, realistic=True)
    model = MujocoModel(spec)
    model.apply_angle_bisector_control()
    mujoco.mj_forward(model.model, model.data)

    for target in model.angle_bisector_controller.targets:
        node_pos = model.data.site_xpos[target.node_site_id]
        neighbor_a = model.data.site_xpos[target.neighbor_site_ids[0]]
        neighbor_b = model.data.site_xpos[target.neighbor_site_ids[1]]
        bisector = _unit(_unit(neighbor_a - node_pos) + _unit(neighbor_b - node_pos))
        planar_bisector = bisector - target.hinge_axis * float(np.dot(bisector, target.hinge_axis))
        planar_bisector = _unit(planar_bisector)

        tip_site_id = mujoco.mj_name2id(
            model.model,
            mujoco.mjtObj.mjOBJ_SITE,
            f"tip_site_{target.node_name}",
        )
        rod_direction = _unit(model.data.site_xpos[tip_site_id] - node_pos)

        assert float(np.dot(rod_direction, planar_bisector)) == pytest.approx(
            -1.0,
            abs=1e-4,
        )
        assert abs(float(np.dot(rod_direction, target.hinge_axis))) < 1e-6


def test_node_velocity_controller_maps_node_commands_to_edge_commands() -> None:
    node_dict = {
        "node_1": [0.0, 0.0, 0.2],
        "node_2": [1.0, 0.0, 0.2],
        "node_3": [2.0, 0.0, 0.2],
    }
    shape_dict = {
        "path_1": {
            "route": ["node_1", "node_2", "node_3"],
            "active_edges": [["node_1", "node_2"]],
        },
    }
    model = MujocoModel(get_mujoco_spec(node_dict, shape_dict, realistic=False))
    controller = NodeVelocityController(
        model.model,
        model.xml,
        model.node_names,
        model.site_to_node,
        model.external_actuator_ids,
    )

    np.testing.assert_array_equal(controller.passive_node_names, ["node_1", "node_3"])
    np.testing.assert_allclose(
        controller.incidence_matrix,
        np.array([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]]),
    )

    edge_commands = controller.transform(np.array([1.0, 2.0, 3.0]))

    np.testing.assert_allclose(controller.latest_node_commands, [0.0, 2.0, 0.0])
    np.testing.assert_allclose(edge_commands, [2.0, -2.0])


def test_node_velocity_controller_clips_edge_commands() -> None:
    node_dict = {
        "node_1": [0.0, 0.0, 0.2],
        "node_2": [1.0, 0.0, 0.2],
        "node_3": [2.0, 0.0, 0.2],
    }
    shape_dict = {
        "path_1": {
            "route": ["node_1", "node_2", "node_3"],
            "active_edges": [["node_1", "node_2"]],
        },
    }
    model = MujocoModel(get_mujoco_spec(node_dict, shape_dict, realistic=False))
    controller = NodeVelocityController(
        model.model,
        model.xml,
        model.node_names,
        model.site_to_node,
        model.external_actuator_ids,
    )

    np.testing.assert_allclose(
        controller.clipped_edge_commands(model.model, np.array([0.0, 2.0, 0.0])),
        [0.05, -0.05],
    )


def test_node_velocity_controller_uses_first_conflicting_route_orientation() -> None:
    node_dict = {
        "node_1": [0.0, 0.0, 0.2],
        "node_2": [1.0, 0.0, 0.2],
    }
    shape_dict = {
        "path_1": {
            "route": ["node_1", "node_2"],
            "active_edges": [["node_1", "node_2"]],
        },
        "path_2": {
            "route": ["node_2", "node_1"],
            "active_edges": [["node_2", "node_1"]],
        },
    }
    model = MujocoModel(get_mujoco_spec(node_dict, shape_dict, realistic=False))
    controller = NodeVelocityController(
        model.model,
        model.xml,
        model.node_names,
        model.site_to_node,
        model.external_actuator_ids,
    )

    assert [(edge.from_node, edge.to_node) for edge in controller.edges] == [
        ("node_1", "node_2")
    ]


def test_node_velocity_command_env_steps_with_node_actions() -> None:
    node_dict = {
        "node_1": [0.0, 0.0, 0.2],
        "node_2": [1.0, 0.0, 0.2],
        "node_3": [2.0, 0.0, 0.2],
    }
    shape_dict = {
        "path_1": {
            "route": ["node_1", "node_2", "node_3"],
            "active_edges": [["node_1", "node_2"]],
        },
    }
    env = MujocoNodeVelocityCommandEnv(
        TrussEnvConfig(
            get_mujoco_spec(node_dict, shape_dict, realistic=False),
            max_steps=2,
            nsubsteps=1,
            speed=0.01,
        )
    )
    try:
        obs, _ = env.reset(seed=13)
        assert env.observation_space.contains(obs)
        assert env.action_space.shape == (3,)

        action = np.array([0.01, 0.02, 0.03], dtype=np.float32)
        obs, _, _, _, info = env.step(action)

        assert env.observation_space.contains(obs)
        assert "critical_eig" in info
        np.testing.assert_allclose(
            env.node_velocity_controller.latest_node_commands,
            [0.0, 0.01, 0.0],
        )
        np.testing.assert_allclose(env.mj_model.get_external_ctrl(), [0.01, -0.01])
    finally:
        env.close()


def test_node_velocity_viewer_state_tracks_sliders_and_tendon_readouts() -> None:
    state = NodeVelocityViewerState(
        ["node_1", "node_2"],
        ["tendon_node_1_node_2"],
        ["node_1"],
        speed=0.01,
    )

    state.set_node_command("node_1", 0.01)
    state.set_node_command("node_2", 0.02)
    state.set_edge_commands(np.array([0.015]))

    np.testing.assert_allclose(state.node_commands, [0.0, 0.01])
    np.testing.assert_allclose(state.edge_commands, [0.015])


def test_node_velocity_terminal_commands_update_node_commands() -> None:
    node_dict = {
        "node_1": [0.0, 0.0, 0.2],
        "node_2": [1.0, 0.0, 0.2],
        "node_3": [2.0, 0.0, 0.2],
    }
    shape_dict = {
        "path_1": {
            "route": ["node_1", "node_2", "node_3"],
            "active_edges": [["node_1", "node_2"]],
        },
    }
    model = MujocoModel(get_mujoco_spec(node_dict, shape_dict, realistic=False))
    controller = NodeVelocityController(
        model.model,
        model.xml,
        model.node_names,
        model.site_to_node,
        model.external_actuator_ids,
    )
    node_commands = np.zeros(len(controller.node_names), dtype=float)

    assert not _apply_terminal_command("set node_2 0.02", controller, node_commands, 0.01)
    np.testing.assert_allclose(node_commands, [0.0, 0.01, 0.0])
    np.testing.assert_allclose(controller.latest_edge_commands, [0.01, -0.01])

    assert not _apply_terminal_command("add 1 -0.005", controller, node_commands, 0.01)
    np.testing.assert_allclose(node_commands, [0.0, 0.005, 0.0])

    assert not _apply_terminal_command("set node_1 0.01", controller, node_commands, 0.01)
    np.testing.assert_allclose(node_commands, [0.0, 0.005, 0.0])

    assert not _apply_terminal_command("zero", controller, node_commands, 0.01)
    np.testing.assert_allclose(node_commands, [0.0, 0.0, 0.0])
    assert _apply_terminal_command("quit", controller, node_commands, 0.01)


def test_realistic_logical_gnn_edge_index_matches_abstract_graph() -> None:
    abstract_model = get_mujoco_spec("octahedron", realistic=False)
    realistic_model = get_mujoco_spec("octahedron", realistic=True)

    abstract_edge_index = get_edge_index(abstract_model)
    realistic_edge_index = get_edge_index(realistic_model, graph_view="logical")

    assert realistic_edge_index.shape == abstract_edge_index.shape
    assert realistic_edge_index.shape == (2, 24)
    assert int(np.max(realistic_edge_index)) < get_node_features(
        realistic_model,
        graph_view="logical",
    ).shape[0]


def test_realistic_logical_gnn_node_features_support_mean_aggregation() -> None:
    model = MujocoModel(get_mujoco_spec("octahedron", realistic=True))

    logical_features = get_node_features(model, graph_view="logical", aggregation="mean")
    physical_positions = model.get_node_position_dict()
    physical_velocities = model.get_node_velocity_linear_dict()
    node_1_instances = [
        node_name
        for node_name in model.node_names
        if node_name == "node_1" or node_name.startswith("node_1_tri_")
    ]
    expected_node_1 = np.concatenate(
        [
            np.mean([physical_positions[node_name] for node_name in node_1_instances], axis=0),
            np.mean([physical_velocities[node_name] for node_name in node_1_instances], axis=0),
        ]
    )

    assert logical_features.shape == (6, 6)
    np.testing.assert_allclose(logical_features[0], expected_node_1)


def test_realistic_logical_gnn_node_features_support_connector_ball_aggregation() -> None:
    node_dict = {
        "node_1": [0.0, 0.0, 0.2],
        "node_2": [0.8, 0.0, 0.2],
        "node_3": [0.4, 0.7, 0.2],
        "node_4": [0.4, -0.7, 0.2],
    }
    triangle_dict = {
        "triangle_1": ["node_1", "node_2", "node_3", "node_1"],
        "triangle_2": ["node_1", "node_4", "node_2", "node_1"],
    }
    model = MujocoModel(get_mujoco_spec(node_dict, triangle_dict, realistic=True))

    logical_features = get_node_features(
        model,
        graph_view="logical",
        aggregation="connector_ball",
    )
    connector_ball_id = mujoco.mj_name2id(
        model.model,
        mujoco.mjtObj.mjOBJ_BODY,
        "connector_ball_node_1",
    )
    node_3_id = model.node_body_ids["node_3"]
    expected_node_1 = np.concatenate(
        [model.data.xpos[connector_ball_id], model.data.cvel[connector_ball_id][3:]]
    )
    expected_node_3 = np.concatenate(
        [model.data.xpos[node_3_id], model.data.cvel[node_3_id][3:]]
    )

    assert logical_features.shape == (4, 6)
    np.testing.assert_allclose(logical_features[0], expected_node_1)
    np.testing.assert_allclose(logical_features[2], expected_node_3)


def test_routed_shape_generation_does_not_mutate_custom_dictionaries() -> None:
    node_dict = {
        "node_1": [0.0, 0.0, 0.2],
        "node_2": [0.8, 0.0, 0.2],
        "node_3": [0.8, 0.8, 0.2],
        "node_4": [0.0, 0.8, 0.2],
    }
    shape_dict = {
        "quad_1": {
            "route": ["node_1", "node_2", "node_3", "node_4", "node_1"],
            "active_edges": [["node_1", "node_2"], ["node_4", "node_1"]],
        },
    }
    original_nodes = deepcopy(node_dict)
    original_shapes = deepcopy(shape_dict)

    get_mujoco_spec(node_dict, shape_dict, realistic=False).compile()

    assert node_dict == original_nodes
    assert shape_dict == original_shapes

    get_mujoco_spec(node_dict, shape_dict, realistic=True).compile()

    assert node_dict == original_nodes
    assert shape_dict == original_shapes


def test_tetrahedron_routed_shape_has_no_route_constraints() -> None:
    spec = get_mujoco_spec("tetrahedron", realistic=False)
    model = spec.compile()
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    assert data.nefc == 0


def test_actuator_names_are_edge_based() -> None:
    model = get_mujoco_spec("tetrahedron", realistic=False).compile()

    actuator_names = {model.actuator(index).name for index in range(model.nu)}

    assert actuator_names == {"act_12", "act_24", "act_34", "act_23", "act_13", "act_14"}


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _xml_vector(value: str) -> np.ndarray:
    return np.fromstring(value, sep=" ", dtype=float)


def _quat_rotate_x(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.array(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
        ]
    )


def _quat_rotate_z(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.array(
        [
            2.0 * (x * z + y * w),
            2.0 * (y * z - x * w),
            1.0 - 2.0 * (x * x + y * y),
        ]
    )
