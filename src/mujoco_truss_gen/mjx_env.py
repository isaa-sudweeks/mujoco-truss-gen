from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import mujoco  # type: ignore[import-untyped]
import numpy as np
from mujoco import mjx

from mujoco_truss_gen.base_env import Range, TrussEnvConfig, _coerce_config
from mujoco_truss_gen.mjx_controllers import MjxAngleBisectorController
from mujoco_truss_gen.mujoco_model.controllers import NodeVelocityController
from mujoco_truss_gen.mujoco_model.model import ModelSource, MujocoModel

MjxInfo = dict[str, jax.Array]


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class MjxDomainRandomizationState:
    """Per-environment domain parameters for :class:`MjxNodeVelocityEnv`."""

    body_mass_multiplier: jax.Array
    body_inertia_multiplier: jax.Array
    dof_damping_multiplier: jax.Array
    dof_armature: jax.Array
    dof_frictionloss: jax.Array
    actuator_gain_multiplier: jax.Array
    actuator_bias_multiplier: jax.Array
    actuator_dynprm_multiplier: jax.Array
    geom_friction_slide: jax.Array
    geom_friction_torsional: jax.Array
    geom_friction_rolling: jax.Array
    tendon_stiffness: jax.Array
    tendon_damping: jax.Array
    tendon_armature: jax.Array
    tendon_frictionloss: jax.Array
    gravity_z: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class MjxEnvState:
    """Batched dynamic state for :class:`MjxNodeVelocityEnv`."""

    data: mjx.Data
    step_count: jax.Array
    node_commands: jax.Array
    domain_randomization: MjxDomainRandomizationState


class MjxNodeVelocityEnv:
    """Pure, batch-native MJX environment with node velocity commands.

    One instance owns a fixed MJX model and task configuration. ``reset``,
    ``step``, and ``reset_where`` accept and return arrays with a leading batch
    dimension and can be compiled by applying :func:`jax.jit` to the bound
    methods.
    """

    def __init__(
        self,
        model_source: TrussEnvConfig | ModelSource,
        **config_overrides: Any,
    ) -> None:
        self.config = _coerce_config(model_source, config_overrides)
        self._validate_config()
        self._domain_randomization = self.config.domain_randomization

        self.mujoco_model = MujocoModel(self.config.model_source)
        model = self.mujoco_model.model
        if (
            self._domain_randomization is not None
            and self._domain_randomization.dof_damping_multiplier_range is not None
            and not np.any(model.dof_damping)
        ):
            warnings.warn(
                "dof_damping_multiplier_range has no effect because the model's "
                "nominal dof_damping array is entirely zero; configure nominal joint "
                "damping or omit this randomization range.",
                UserWarning,
                stacklevel=2,
            )
        self._angle_bisector_controller = MjxAngleBisectorController(
            self.mujoco_model.angle_bisector_controller.targets
        )
        unsupported_internal_actuators = (
            set(self.mujoco_model.internal_actuator_ids.tolist())
            - self._angle_bisector_controller.actuator_ids
        )
        if unsupported_internal_actuators:
            names = [
                model.actuator(actuator_id).name for actuator_id in unsupported_internal_actuators
            ]
            raise ValueError(
                "MjxNodeVelocityEnv does not support internal actuator(s) not owned by "
                f"the angle-bisector controller: {', '.join(sorted(names))}."
            )

        self._controller = NodeVelocityController(
            model,
            self.mujoco_model.xml,
            self.mujoco_model.node_names,
            self.mujoco_model.site_to_node,
            self.mujoco_model.external_actuator_ids,
        )
        if not self._controller.enabled:
            raise ValueError(
                "MjxNodeVelocityEnv requires model control-graph metadata and "
                "node-routed tendon actuators."
            )

        try:
            self.mjx_model = mjx.put_model(model)
            self._data_template = mjx.put_data(model, self.mujoco_model.data)
        except (NotImplementedError, ValueError) as error:
            raise ValueError(f"Model is not compatible with MJX: {error}") from error
        self._enable_randomized_frictionloss_constraints()
        const_data_template = mjx.forward(self.mjx_model, self._data_template)
        self._tendon_jacobian0 = const_data_template._impl.ten_J
        self._actuator_moment0 = const_data_template._impl.actuator_moment
        # MJX stores qM densely for small models and in MuJoCo's packed sparse
        # representation for larger models.  The domain-randomization helpers
        # below operate on the full matrix in either case.
        self._qM0 = mjx.full_m(self.mjx_model, const_data_template)
        self._nominal_qm_physical = self._nominal_physical_mass_matrix()
        self._actuator_acc0_scale = self._nominal_actuator_acc0_scale()

        self.action_size = len(self._controller.node_names)
        self.observation_size = 7 * self.action_size
        self.action_low = jnp.full((self.action_size,), -float(self.config.speed))
        self.action_high = jnp.full((self.action_size,), float(self.config.speed))

        self._domain_dtype = self.mjx_model.body_mass.dtype
        self._nominal_gravity_z = float(model.opt.gravity[2])
        self._body_mass_multiplier_range = self._jax_range(
            self._domain_randomization.body_mass_multiplier_range
            if self._domain_randomization is not None
            else None
        )
        self._body_inertia_multiplier_range = self._jax_range(
            self._domain_randomization.body_inertia_multiplier_range
            if self._domain_randomization is not None
            else None
        )
        self._dof_damping_multiplier_range = self._jax_range(
            self._domain_randomization.dof_damping_multiplier_range
            if self._domain_randomization is not None
            else None
        )
        self._dof_armature_range = self._jax_range(
            self._domain_randomization.dof_armature_range
            if self._domain_randomization is not None
            else None
        )
        self._dof_frictionloss_range = self._jax_range(
            self._domain_randomization.dof_frictionloss_range
            if self._domain_randomization is not None
            else None
        )
        self._actuator_gain_multiplier_range = self._jax_range(
            self._domain_randomization.actuator_gain_multiplier_range
            if self._domain_randomization is not None
            else None
        )
        self._actuator_bias_multiplier_range = self._jax_range(
            self._domain_randomization.actuator_bias_multiplier_range
            if self._domain_randomization is not None
            else None
        )
        self._actuator_dynprm_multiplier_range = self._jax_range(
            self._domain_randomization.actuator_dynprm_multiplier_range
            if self._domain_randomization is not None
            else None
        )
        self._geom_friction_slide_range = self._jax_range(
            self._domain_randomization.geom_friction_slide_range
            if self._domain_randomization is not None
            else None
        )
        self._geom_friction_torsional_range = self._jax_range(
            self._domain_randomization.geom_friction_torsional_range
            if self._domain_randomization is not None
            else None
        )
        self._geom_friction_rolling_range = self._jax_range(
            self._domain_randomization.geom_friction_rolling_range
            if self._domain_randomization is not None
            else None
        )
        self._tendon_stiffness_range = self._jax_range(
            self._domain_randomization.tendon_stiffness_range
            if self._domain_randomization is not None
            else None
        )
        self._tendon_damping_range = self._jax_range(
            self._domain_randomization.tendon_damping_range
            if self._domain_randomization is not None
            else None
        )
        self._tendon_armature_range = self._jax_range(
            self._domain_randomization.tendon_armature_range
            if self._domain_randomization is not None
            else None
        )
        self._tendon_frictionloss_range = self._jax_range(
            self._domain_randomization.tendon_frictionloss_range
            if self._domain_randomization is not None
            else None
        )
        self._gravity_z_range = self._jax_range(
            self._domain_randomization.gravity_z_range
            if self._domain_randomization is not None
            else None
        )

        self._passive_node_mask = jnp.asarray(self._controller.passive_node_mask)
        self._incidence_matrix = jnp.asarray(self._controller.incidence_matrix)
        self._actuator_ids = jnp.asarray(self._controller.actuator_ids, dtype=jnp.int32)
        ctrlrange = model.actuator_ctrlrange[self._controller.actuator_ids]
        self._ctrl_low = jnp.asarray(ctrlrange[:, 0])
        self._ctrl_high = jnp.asarray(ctrlrange[:, 1])

        control_graph = self.mujoco_model.control_graph
        self._control_body_ids = jnp.asarray(
            [
                self.mujoco_model.node_body_ids[
                    control_graph.control_node_to_physical_node[node_name]
                ]
                for node_name in self._controller.node_names
            ],
            dtype=jnp.int32,
        )
        self._node_body_ids = jnp.asarray(
            [
                self.mujoco_model.node_body_ids[node_name]
                for node_name in self.mujoco_model.node_names
            ],
            dtype=jnp.int32,
        )
        self._bbox_dimensions = jnp.asarray(self.mujoco_model.initial_bounding_box_dimensions)
        self._position_scale = float(max(self.mujoco_model.initial_bounding_box_diagonal, 1e-8))
        self._initial_critical_eig = float(self.mujoco_model.initial_critical_eig)

        if self.mujoco_model._uses_realistic_connector_balls():
            rigidity_node_names, _, rigidity_edges_by_name, rigidity_axis_indices = (
                self.mujoco_model._logical_rigidity_graph()
            )
        else:
            rigidity_node_names = self.mujoco_model.node_names
            rigidity_edges_by_name = self.mujoco_model.structural_edges
            rigidity_axis_indices = self.mujoco_model.axis_indices

        rigidity_node_index = {
            node_name: index for index, node_name in enumerate(rigidity_node_names)
        }
        rigidity_edges = [
            (rigidity_node_index[node_a], rigidity_node_index[node_b])
            for node_a, node_b in rigidity_edges_by_name
            if node_a in rigidity_node_index and node_b in rigidity_node_index and node_a != node_b
        ]
        self._rigidity_edge_a = jnp.asarray([edge[0] for edge in rigidity_edges], dtype=jnp.int32)
        self._rigidity_edge_b = jnp.asarray([edge[1] for edge in rigidity_edges], dtype=jnp.int32)
        self._axis_indices = jnp.asarray(rigidity_axis_indices, dtype=jnp.int32)
        self._rigidity_body_ids, self._rigidity_body_mask = self._rigidity_body_metadata(
            rigidity_node_names
        )

        reset_actuator_ids = np.array(
            [
                actuator_id
                for actuator_id in range(model.nu)
                if model.actuator_trntype[actuator_id] == mujoco.mjtTrn.mjTRN_TENDON
                and model.actuator_dyntype[actuator_id] == mujoco.mjtDyn.mjDYN_INTEGRATOR
                and model.actuator_actadr[actuator_id] >= 0
            ],
            dtype=int,
        )
        self._reset_act_adrs = jnp.asarray(
            model.actuator_actadr[reset_actuator_ids], dtype=jnp.int32
        )
        self._reset_tendon_ids = jnp.asarray(
            model.actuator_trnid[reset_actuator_ids, 0], dtype=jnp.int32
        )

    def _enable_randomized_frictionloss_constraints(self) -> None:
        if self._domain_randomization is None:
            return

        impl = self.mjx_model._impl
        if self._domain_randomization.dof_frictionloss_range is not None:
            impl = impl.replace(
                dof_hasfrictionloss=np.ones_like(impl.dof_hasfrictionloss, dtype=bool)
            )
        if self._domain_randomization.tendon_frictionloss_range is not None:
            impl = impl.replace(
                tendon_hasfrictionloss=np.ones_like(
                    impl.tendon_hasfrictionloss, dtype=bool
                )
            )
        self.mjx_model = self.mjx_model.replace(_impl=impl)

    def _nominal_physical_mass_matrix(self) -> jax.Array:
        tendon_armature_diag = jnp.sum(
            jnp.square(self._tendon_jacobian0) * self.mjx_model.tendon_armature[:, None],
            axis=0,
        )
        return (
            self._qM0
            - jnp.diag(self.mjx_model.dof_armature)
            - jnp.diag(tendon_armature_diag)
        )

    def _nominal_actuator_acc0_scale(self) -> jax.Array:
        nominal_inverse_qm = jnp.linalg.inv(self._qM0)
        nominal_quadratic = jnp.einsum(
            "ij,jk,ik->i",
            self._actuator_moment0,
            nominal_inverse_qm,
            self._actuator_moment0,
        )
        return jnp.where(
            nominal_quadratic > 0.0,
            self.mjx_model.actuator_acc0 / nominal_quadratic,
            0.0,
        )

    def _validate_config(self) -> None:
        randomization = self.config.domain_randomization
        if randomization is not None:
            if randomization.model_factory is not None:
                raise ValueError(
                    "MjxNodeVelocityEnv requires one fixed model shape; "
                    "DomainRandomizationConfig.model_factory is not supported."
                )
            for name in (
                "body_mass_multiplier_range",
                "body_inertia_multiplier_range",
                "dof_damping_multiplier_range",
                "dof_armature_range",
                "dof_frictionloss_range",
                "actuator_gain_multiplier_range",
                "actuator_bias_multiplier_range",
                "actuator_dynprm_multiplier_range",
                "geom_friction_slide_range",
                "geom_friction_torsional_range",
                "geom_friction_rolling_range",
                "tendon_stiffness_range",
                "tendon_damping_range",
                "tendon_armature_range",
                "tendon_frictionloss_range",
                "gravity_z_range",
            ):
                self._validate_range(getattr(randomization, name), name)
        if int(self.config.max_steps) <= 0:
            raise ValueError("max_steps must be greater than zero.")
        if int(self.config.nsubsteps) <= 0:
            raise ValueError("nsubsteps must be greater than zero.")
        if not np.isfinite(self.config.speed) or float(self.config.speed) < 0.0:
            raise ValueError("speed must be finite and non-negative.")

    def reset(self, keys: jax.Array) -> tuple[jax.Array, MjxEnvState]:
        """Reset a batch from one explicit random key per environment."""

        batch_size = self._key_batch_size(keys)
        split_keys = jax.vmap(lambda key: jax.random.split(key))(keys)
        reset_keys = split_keys[:, 0]
        domain_keys = split_keys[:, 1]
        domain_randomization = jax.vmap(self._sample_domain_randomization)(domain_keys)
        data = jax.vmap(self._reset_one)(reset_keys, domain_randomization)
        state = MjxEnvState(
            data=data,
            step_count=jnp.zeros((batch_size,), dtype=jnp.int32),
            node_commands=jnp.zeros((batch_size, self.action_size), dtype=self.action_low.dtype),
            domain_randomization=domain_randomization,
        )
        return self._get_obs(state), state

    def step(
        self,
        keys: jax.Array,
        state: MjxEnvState,
        actions: jax.Array,
    ) -> tuple[jax.Array, MjxEnvState, jax.Array, jax.Array, MjxInfo]:
        """Advance every environment by one control step."""

        batch_size = self._key_batch_size(keys)
        self._validate_state_and_action_shapes(state, actions, batch_size)
        return jax.vmap(self._step_one)(keys, state, actions)

    def reset_where(
        self,
        keys: jax.Array,
        state: MjxEnvState,
        mask: jax.Array,
    ) -> tuple[jax.Array, MjxEnvState]:
        """Reset selected batch elements while preserving all others."""

        batch_size = self._key_batch_size(keys)
        if mask.shape != (batch_size,):
            raise ValueError(f"mask must have shape ({batch_size},), got {mask.shape}.")
        if state.step_count.shape != (batch_size,):
            raise ValueError("state batch dimension must match the number of reset keys.")

        _, reset_state = self.reset(keys)
        mask = jnp.asarray(mask, dtype=jnp.bool_)

        def select(reset_value: jax.Array, old_value: jax.Array) -> jax.Array:
            expanded_mask = mask.reshape((batch_size,) + (1,) * (reset_value.ndim - 1))
            return jnp.where(expanded_mask, reset_value, old_value)

        merged_state = jax.tree.map(select, reset_state, state)
        return self._get_obs(merged_state), merged_state

    def _reset_one(
        self,
        key: jax.Array,
        domain_randomization: MjxDomainRandomizationState,
    ) -> mjx.Data:
        qpos_key, qvel_key = jax.random.split(key)
        model = self._model_for_domain(domain_randomization)
        data = self._data_template.replace(
            qpos=self._data_template.qpos
            + jax.random.uniform(
                qpos_key,
                self._data_template.qpos.shape,
                minval=-0.005,
                maxval=0.005,
            ),
            qvel=self._data_template.qvel
            + jax.random.uniform(
                qvel_key,
                self._data_template.qvel.shape,
                minval=-0.005,
                maxval=0.005,
            ),
            ctrl=jnp.zeros_like(self._data_template.ctrl),
        )
        data = mjx.forward(model, data)
        data = self._angle_bisector_controller.initialize(data)
        if self._reset_act_adrs.size:
            act = data.act.at[self._reset_act_adrs].set(data.ten_length[self._reset_tendon_ids])
            data = data.replace(act=act)
        return mjx.forward(model, data)

    def _step_one(
        self,
        key: jax.Array,
        state: MjxEnvState,
        action: jax.Array,
    ) -> tuple[jax.Array, MjxEnvState, jax.Array, jax.Array, MjxInfo]:
        action = jnp.clip(action, self.action_low, self.action_high)
        model = self._model_for_domain(state.domain_randomization)
        previous_com = self._center_of_mass(state.data)

        node_commands = jnp.where(self._passive_node_mask, 0.0, action)
        edge_commands = self._incidence_matrix @ node_commands
        edge_commands = jnp.clip(edge_commands, self._ctrl_low, self._ctrl_high)
        edge_commands = self._apply_control_noise(key, edge_commands)

        ctrl = state.data.ctrl.at[self._actuator_ids].set(edge_commands)
        data = state.data.replace(ctrl=ctrl)

        def physics_substep(_index: int, loop_data: mjx.Data) -> mjx.Data:
            loop_data = self._angle_bisector_controller.update(loop_data)
            return mjx.step(model, loop_data)

        data = jax.lax.fori_loop(0, int(self.config.nsubsteps), physics_substep, data)

        step_count = state.step_count + jnp.asarray(1, dtype=state.step_count.dtype)
        next_state = MjxEnvState(
            data=data,
            step_count=step_count,
            node_commands=node_commands,
            domain_randomization=state.domain_randomization,
        )
        reward, info, terminated = self._compute_reward(data, action, previous_com)
        truncated = step_count >= int(self.config.max_steps)
        done = jnp.logical_or(terminated, truncated)
        info = dict(info)
        info["terminated"] = terminated
        info["truncated"] = truncated
        return self._get_obs_one(data, node_commands), next_state, reward, done, info

    def _sample_domain_randomization(self, key: jax.Array) -> MjxDomainRandomizationState:
        keys = jax.random.split(key, 16)
        return MjxDomainRandomizationState(
            body_mass_multiplier=self._sample_jax_range(
                keys[0], self._body_mass_multiplier_range, 1.0
            ),
            body_inertia_multiplier=self._sample_jax_range(
                keys[1], self._body_inertia_multiplier_range, 1.0
            ),
            dof_damping_multiplier=self._sample_jax_range(
                keys[2], self._dof_damping_multiplier_range, 1.0
            ),
            dof_armature=self._sample_jax_range(keys[3], self._dof_armature_range, 0.0),
            dof_frictionloss=self._sample_jax_range(keys[4], self._dof_frictionloss_range, 0.0),
            actuator_gain_multiplier=self._sample_jax_range(
                keys[5], self._actuator_gain_multiplier_range, 1.0
            ),
            actuator_bias_multiplier=self._sample_jax_range(
                keys[6], self._actuator_bias_multiplier_range, 1.0
            ),
            actuator_dynprm_multiplier=self._sample_jax_range(
                keys[7], self._actuator_dynprm_multiplier_range, 1.0
            ),
            geom_friction_slide=self._sample_jax_range(
                keys[8], self._geom_friction_slide_range, self.mjx_model.geom_friction[0, 0]
            ),
            geom_friction_torsional=self._sample_jax_range(
                keys[9],
                self._geom_friction_torsional_range,
                self.mjx_model.geom_friction[0, 1],
            ),
            geom_friction_rolling=self._sample_jax_range(
                keys[10],
                self._geom_friction_rolling_range,
                self.mjx_model.geom_friction[0, 2],
            ),
            tendon_stiffness=self._sample_jax_range(
                keys[11], self._tendon_stiffness_range, 0.0
            ),
            tendon_damping=self._sample_jax_range(keys[12], self._tendon_damping_range, 0.0),
            tendon_armature=self._sample_jax_range(keys[13], self._tendon_armature_range, 0.0),
            tendon_frictionloss=self._sample_jax_range(
                keys[14], self._tendon_frictionloss_range, 0.0
            ),
            gravity_z=self._sample_jax_range(
                keys[15], self._gravity_z_range, self._nominal_gravity_z
            ),
        )

    def _nominal_domain_randomization_state(self) -> MjxDomainRandomizationState:
        dtype = self._domain_dtype
        return MjxDomainRandomizationState(
            body_mass_multiplier=jnp.asarray(1.0, dtype=dtype),
            body_inertia_multiplier=jnp.asarray(1.0, dtype=dtype),
            dof_damping_multiplier=jnp.asarray(1.0, dtype=dtype),
            dof_armature=jnp.asarray(0.0, dtype=dtype),
            dof_frictionloss=jnp.asarray(0.0, dtype=dtype),
            actuator_gain_multiplier=jnp.asarray(1.0, dtype=dtype),
            actuator_bias_multiplier=jnp.asarray(1.0, dtype=dtype),
            actuator_dynprm_multiplier=jnp.asarray(1.0, dtype=dtype),
            geom_friction_slide=jnp.asarray(self.mjx_model.geom_friction[0, 0], dtype=dtype),
            geom_friction_torsional=jnp.asarray(
                self.mjx_model.geom_friction[0, 1], dtype=dtype
            ),
            geom_friction_rolling=jnp.asarray(self.mjx_model.geom_friction[0, 2], dtype=dtype),
            tendon_stiffness=jnp.asarray(0.0, dtype=dtype),
            tendon_damping=jnp.asarray(0.0, dtype=dtype),
            tendon_armature=jnp.asarray(0.0, dtype=dtype),
            tendon_frictionloss=jnp.asarray(0.0, dtype=dtype),
            gravity_z=jnp.asarray(self._nominal_gravity_z, dtype=dtype),
        )

    def _model_for_domain(self, domain: MjxDomainRandomizationState) -> mjx.Model:
        model = self.mjx_model

        if self._body_mass_multiplier_range is not None:
            multiplier = domain.body_mass_multiplier
            inverse_multiplier = 1.0 / multiplier
            model = model.replace(
                body_mass=model.body_mass * multiplier,
                body_subtreemass=model.body_subtreemass * multiplier,
                body_invweight0=model.body_invweight0 * inverse_multiplier,
                dof_invweight0=model.dof_invweight0 * inverse_multiplier,
                dof_M0=model.dof_M0 * multiplier,
                tendon_invweight0=model.tendon_invweight0 * inverse_multiplier,
                actuator_acc0=model.actuator_acc0 * inverse_multiplier,
            )
        if self._body_inertia_multiplier_range is not None:
            model = model.replace(body_inertia=model.body_inertia * domain.body_inertia_multiplier)
        if self._dof_damping_multiplier_range is not None:
            model = model.replace(dof_damping=model.dof_damping * domain.dof_damping_multiplier)
        if self._dof_armature_range is not None:
            model = model.replace(
                dof_armature=jnp.full_like(model.dof_armature, domain.dof_armature),
                dof_M0=model.dof_M0
                + domain.dof_armature
                - self.mjx_model.dof_armature * domain.body_mass_multiplier,
            )
        if self._dof_frictionloss_range is not None:
            model = model.replace(
                dof_frictionloss=jnp.full_like(model.dof_frictionloss, domain.dof_frictionloss)
            )
        if self._actuator_gain_multiplier_range is not None:
            model = model.replace(
                actuator_gainprm=model.actuator_gainprm * domain.actuator_gain_multiplier
            )
        if self._actuator_bias_multiplier_range is not None:
            model = model.replace(
                actuator_biasprm=model.actuator_biasprm * domain.actuator_bias_multiplier
            )
        if self._actuator_dynprm_multiplier_range is not None:
            model = model.replace(
                actuator_dynprm=model.actuator_dynprm * domain.actuator_dynprm_multiplier
            )
        if self._geom_friction_slide_range is not None:
            model = model.replace(
                geom_friction=model.geom_friction.at[:, 0].set(domain.geom_friction_slide)
            )
        if self._geom_friction_torsional_range is not None:
            model = model.replace(
                geom_friction=model.geom_friction.at[:, 1].set(
                    domain.geom_friction_torsional
                )
            )
        if self._geom_friction_rolling_range is not None:
            model = model.replace(
                geom_friction=model.geom_friction.at[:, 2].set(domain.geom_friction_rolling)
            )
        if self._tendon_stiffness_range is not None:
            model = model.replace(
                tendon_stiffness=jnp.full_like(model.tendon_stiffness, domain.tendon_stiffness)
            )
        if self._tendon_damping_range is not None:
            model = model.replace(
                tendon_damping=jnp.full_like(model.tendon_damping, domain.tendon_damping)
            )
        if self._tendon_armature_range is not None:
            model = model.replace(
                tendon_armature=jnp.full_like(model.tendon_armature, domain.tendon_armature)
            )
        if self._tendon_frictionloss_range is not None:
            model = model.replace(
                tendon_frictionloss=jnp.full_like(
                    model.tendon_frictionloss, domain.tendon_frictionloss
                )
            )
        if (
            self._body_mass_multiplier_range is not None
            or self._dof_armature_range is not None
            or self._tendon_armature_range is not None
        ):
            inverse_qm = self._reference_inverse_mass_matrix_for_domain(domain)
            model = model.replace(
                tendon_invweight0=self._tendon_invweight0_for_domain(inverse_qm),
                actuator_acc0=self._actuator_acc0_for_domain(inverse_qm),
            )
        if self._gravity_z_range is not None:
            model = model.replace(
                opt=model.opt.replace(gravity=model.opt.gravity.at[2].set(domain.gravity_z))
            )
        return model

    def _reference_inverse_mass_matrix_for_domain(
        self,
        domain: MjxDomainRandomizationState,
    ) -> jax.Array:
        mass_multiplier = (
            domain.body_mass_multiplier if self._body_mass_multiplier_range is not None else 1.0
        )
        dof_armature = (
            jnp.full_like(self.mjx_model.dof_armature, domain.dof_armature)
            if self._dof_armature_range is not None
            else self.mjx_model.dof_armature
        )
        tendon_armature = (
            jnp.full_like(self.mjx_model.tendon_armature, domain.tendon_armature)
            if self._tendon_armature_range is not None
            else self.mjx_model.tendon_armature
        )
        tendon_armature_diag = jnp.sum(
            jnp.square(self._tendon_jacobian0) * tendon_armature[:, None],
            axis=0,
        )
        qm = (
            self._nominal_qm_physical * mass_multiplier
            + jnp.diag(dof_armature)
            + jnp.diag(tendon_armature_diag)
        )
        return jnp.linalg.inv(qm)

    def _tendon_invweight0_for_domain(
        self,
        inverse_qm: jax.Array,
    ) -> jax.Array:
        return jnp.einsum("ij,jk,ik->i", self._tendon_jacobian0, inverse_qm, self._tendon_jacobian0)

    def _actuator_acc0_for_domain(self, inverse_qm: jax.Array) -> jax.Array:
        actuator_quadratic = jnp.einsum(
            "ij,jk,ik->i",
            self._actuator_moment0,
            inverse_qm,
            self._actuator_moment0,
        )
        return actuator_quadratic * self._actuator_acc0_scale

    def _sample_jax_range(
        self,
        key: jax.Array,
        value_range: tuple[jax.Array, jax.Array] | None,
        default: float | jax.Array,
    ) -> jax.Array:
        if value_range is None:
            return jnp.asarray(default, dtype=self._domain_dtype)
        low, high = value_range
        return jax.random.uniform(key, (), minval=low, maxval=high, dtype=self._domain_dtype)

    def _jax_range(self, value_range: Range | None) -> tuple[jax.Array, jax.Array] | None:
        if value_range is None:
            return None
        low, high = value_range
        return (
            jnp.asarray(float(low), dtype=self._domain_dtype),
            jnp.asarray(float(high), dtype=self._domain_dtype),
        )

    @staticmethod
    def _validate_range(value_range: Range | None, name: str) -> None:
        if value_range is None:
            return
        low, high = (float(value_range[0]), float(value_range[1]))
        if not np.isfinite(low) or not np.isfinite(high) or low > high:
            raise ValueError(f"{name} must contain finite values with low <= high.")

    def _apply_control_noise(
        self,
        key: jax.Array,
        edge_commands: jax.Array,
    ) -> jax.Array:
        if not (
            bool(self.config.runtime_apply_control_noise)
            and float(self.config.control_noise_std) > 0.0
        ):
            return edge_commands

        if self.config.control_noise_relative:
            noise_scale = (self._ctrl_high - self._ctrl_low) * float(self.config.control_noise_std)
        else:
            noise_scale = jnp.full_like(edge_commands, float(self.config.control_noise_std))
        noisy_commands = edge_commands + jax.random.normal(key, edge_commands.shape) * noise_scale
        return jnp.clip(noisy_commands, self._ctrl_low, self._ctrl_high)

    def _get_obs(self, state: MjxEnvState) -> jax.Array:
        return jax.vmap(self._get_obs_one)(state.data, state.node_commands)

    def _get_obs_one(
        self,
        data: mjx.Data,
        node_commands: jax.Array,
    ) -> jax.Array:
        positions = data.xpos[self._control_body_ids]
        velocities = data.cvel[self._control_body_ids, 3:]
        com = jnp.mean(positions, axis=0)
        relative_positions = positions.at[:, :2].add(-com[:2])
        if self.config.normalize_observations:
            relative_positions = relative_positions / self._bbox_dimensions
            velocities = velocities / self._bbox_dimensions
        return jnp.concatenate(
            (
                relative_positions.reshape(-1),
                velocities.reshape(-1),
                node_commands,
            )
        ).astype(jnp.float32)

    def _center_of_mass(self, data: mjx.Data) -> jax.Array:
        return jnp.mean(data.xpos[self._node_body_ids], axis=0)

    def _compute_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        previous_com: jax.Array,
    ) -> tuple[jax.Array, MjxInfo, jax.Array]:
        critical_eig_raw = self._critical_eig(data)
        terminated = jnp.logical_or(
            jnp.logical_not(jnp.isfinite(critical_eig_raw)),
            critical_eig_raw < float(self.config.critical_eig_threshold),
        )
        critical_eig = jnp.where(jnp.isfinite(critical_eig_raw), critical_eig_raw, 0.0)

        current_com = self._center_of_mass(data)
        com_delta_x = current_com[0] - previous_com[0]
        dt = float(self.config.nsubsteps) * float(self.mujoco_model.model.opt.timestep)
        raw_forward_vel = jnp.where(dt > 0.0, com_delta_x / dt, 0.0)
        reward_forward_vel = jnp.where(jnp.isfinite(raw_forward_vel), raw_forward_vel, 0.0)
        normalized_forward_vel_raw = reward_forward_vel / self._position_scale
        if self.config.max_forward_velocity is None:
            normalized_forward_vel = normalized_forward_vel_raw
        else:
            velocity_limit = abs(float(self.config.max_forward_velocity))
            normalized_forward_vel = jnp.clip(
                normalized_forward_vel_raw, -velocity_limit, velocity_limit
            )
        if self.config.zero_positive_forward_reward_on_termination:
            normalized_forward_vel = jnp.where(
                terminated, jnp.minimum(normalized_forward_vel, 0.0), normalized_forward_vel
            )
        forward_vel = normalized_forward_vel * self._position_scale

        energy_penalty = jnp.sum(jnp.square(action))
        slip_penalty = self._slip_penalty(data)
        slip_penalty = jnp.where(jnp.isfinite(slip_penalty), slip_penalty, 0.0)
        if self.config.zero_velocity_shaping_on_termination:
            slip_penalty = jnp.where(terminated, 0.0, slip_penalty)

        forward_reward = float(self.config.forward_weight) * normalized_forward_vel
        energy_reward = -float(self.config.energy_weight) * energy_penalty
        rigidity_reward = float(self.config.rigidity_weight) * critical_eig
        if self.config.zero_rigidity_reward_on_termination:
            rigidity_reward = jnp.where(terminated, 0.0, rigidity_reward)
        slip_reward = -float(self.config.slip_weight) * slip_penalty
        alive_reward = jnp.asarray(float(self.config.alive_bonus))
        if self.config.zero_alive_bonus_on_termination:
            alive_reward = jnp.where(terminated, 0.0, alive_reward)
        collapse_penalty = jnp.where(terminated, -abs(float(self.config.collapse_penalty)), 0.0)
        total_reward = (
            forward_reward
            + alive_reward
            + energy_reward
            + rigidity_reward
            + slip_reward
            + collapse_penalty
        )
        info = {
            "forward": forward_reward,
            "forward_velocity": forward_vel,
            "forward_velocity_raw": raw_forward_vel,
            "forward_velocity_normalized": normalized_forward_vel,
            "forward_velocity_normalized_raw": normalized_forward_vel_raw,
            "com_delta_x": com_delta_x,
            "alive": alive_reward,
            "energy": energy_reward,
            "rigidity": rigidity_reward,
            "slip": slip_reward,
            "critical_eig": critical_eig,
            "critical_eig_raw": critical_eig_raw,
            "collapse_penalty": collapse_penalty,
            "terminated_by_collapse": terminated,
        }
        return total_reward, info, terminated

    def _critical_eig(self, data: mjx.Data) -> jax.Array:
        dims = int(self._axis_indices.size)
        node_count = int(self._rigidity_body_ids.shape[0])
        edge_count = int(self._rigidity_edge_a.size)
        rigid_body_modes = dims + (dims * (dims - 1)) // 2
        matrix_width = node_count * dims
        if edge_count == 0 or matrix_width <= rigid_body_modes:
            return jnp.asarray(0.0)

        body_positions = data.xpos[self._rigidity_body_ids]
        mask = self._rigidity_body_mask[..., None]
        positions = jnp.sum(jnp.where(mask, body_positions, 0.0), axis=1)
        positions = positions / jnp.maximum(jnp.sum(mask, axis=1), 1.0)
        positions = positions[:, self._axis_indices]
        delta = positions[self._rigidity_edge_b] - positions[self._rigidity_edge_a]
        lengths = jnp.linalg.norm(delta, axis=1)
        degenerate = lengths < 1e-8
        safe_lengths = jnp.where(degenerate, 1.0, lengths)
        directions = jnp.where(
            degenerate[:, None],
            0.0,
            delta / safe_lengths[:, None],
        )

        rows = jnp.arange(edge_count, dtype=jnp.int32)[:, None]
        axis_offsets = jnp.arange(dims, dtype=jnp.int32)[None, :]
        columns_a = self._rigidity_edge_a[:, None] * dims + axis_offsets
        columns_b = self._rigidity_edge_b[:, None] * dims + axis_offsets
        rigidity = jnp.zeros((edge_count, matrix_width), dtype=positions.dtype)
        rigidity = rigidity.at[rows, columns_a].set(-directions)
        rigidity = rigidity.at[rows, columns_b].set(directions)
        eigvals = jnp.linalg.eigvalsh(rigidity.T @ rigidity)
        raw = jnp.maximum(eigvals[rigid_body_modes], 0.0)
        return raw / self._initial_critical_eig

    def _slip_penalty(self, data: mjx.Data) -> jax.Array:
        positions = data.xpos[self._node_body_ids]
        velocities = data.cvel[self._node_body_ids, 3:]
        contact_mask = positions[:, 2] < float(self.config.slip_height)
        return jnp.sum(jnp.where(contact_mask, jnp.abs(velocities[:, 0]), 0.0))

    def _rigidity_body_metadata(
        self,
        rigidity_node_names: list[str],
    ) -> tuple[jax.Array, jax.Array]:
        model = self.mujoco_model.model
        body_ids_by_node: list[list[int]] = []
        for logical_name in rigidity_node_names:
            connector_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                f"connector_ball_{logical_name}",
            )
            if connector_id >= 0:
                body_ids_by_node.append([connector_id])
                continue

            physical_ids = [
                self.mujoco_model.node_body_ids[node_name]
                for node_name in self.mujoco_model.node_names
                if self.mujoco_model._logical_node_name(node_name) == logical_name
            ]
            body_ids_by_node.append(physical_ids)

        max_instances = max(1, *(len(body_ids) for body_ids in body_ids_by_node))
        body_ids = np.zeros((len(body_ids_by_node), max_instances), dtype=np.int32)
        body_mask = np.zeros_like(body_ids, dtype=bool)
        for index, node_body_ids in enumerate(body_ids_by_node):
            body_ids[index, : len(node_body_ids)] = node_body_ids
            body_mask[index, : len(node_body_ids)] = True
        return jnp.asarray(body_ids), jnp.asarray(body_mask)

    @staticmethod
    def _key_batch_size(keys: jax.Array) -> int:
        if keys.ndim == 0:
            raise ValueError("keys must have a leading batch dimension.")
        return int(keys.shape[0])

    def _validate_state_and_action_shapes(
        self,
        state: MjxEnvState,
        actions: jax.Array,
        batch_size: int,
    ) -> None:
        expected_action_shape = (batch_size, self.action_size)
        if actions.shape != expected_action_shape:
            raise ValueError(
                f"actions must have shape {expected_action_shape}, got {actions.shape}."
            )
        if state.step_count.shape != (batch_size,):
            raise ValueError("state batch dimension must match the number of step keys.")
        if state.node_commands.shape != expected_action_shape:
            raise ValueError("state node-command shape must match the action batch shape.")
        for value in jax.tree.leaves(state.domain_randomization):
            if value.shape != (batch_size,):
                raise ValueError(
                    "state domain-randomization shape must match the number of step keys."
                )
