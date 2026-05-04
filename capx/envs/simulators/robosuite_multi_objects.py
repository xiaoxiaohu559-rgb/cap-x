"""Low-level Robosuite Franka environment with configurable objects.

Supports multiple object types (box, ball, cylinder, bottle, can, etc.)
with arbitrary colours on the table. Generalises the cube-only
``robosuite_multi_cubes.py`` to arbitrary shapes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import viser.transforms as vtf
from robosuite.controllers.composite.composite_controller_factory import (
    load_composite_controller_config,
)
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import (
    BallObject,
    BoxObject,
    CylinderObject,
    MujocoObject,
)
from robosuite.models.objects.xml_objects import (
    BottleObject,
    BreadObject,
    CanObject,
    CerealObject,
    LemonObject,
    MilkObject,
)
from robosuite.models.objects.composite.cone import ConeObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat

from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv

# ── Colour palette ──────────────────────────────────────────────────────────

COLOUR_PALETTE: dict[str, tuple[list[float], str]] = {
    "red":    ([1, 0, 0, 1],    "WoodRed"),
    "green":  ([0, 1, 0, 1],    "WoodGreen"),
    "blue":   ([0, 0, 1, 1],    "WoodBlue"),
    "orange": ([1, 0.5, 0, 1],  "WoodLight"),
    "yellow": ([1, 1, 0, 1],    "WoodLight"),
    "black":  ([0.1, 0.1, 0.1, 1], "WoodDark"),
    "white":  ([0.95, 0.95, 0.95, 1], "WoodLight"),
    "pink":   ([1, 0.4, 0.7, 1], "WoodLight"),
    "purple": ([0.6, 0.2, 0.8, 1], "WoodDark"),
    "gray":   ([0.5, 0.5, 0.5, 1], "WoodLight"),
    "brown":  ([0.55, 0.27, 0.07, 1], "WoodDark"),
    "cyan":   ([0, 0.8, 0.8, 1], "WoodLight"),
}

_DEFAULT_SIZE: dict[str, list[float]] = {
    "box": [0.025, 0.025, 0.025],
    "ball": [0.025],
    "cylinder": [0.02, 0.04],
}

# Estimated bounding-box half-sizes for XML objects (metres).
_XML_BBOX_HALF: dict[str, list[float]] = {
    "bottle":  [0.03, 0.03, 0.06],
    "can":     [0.03, 0.03, 0.05],
    "milk":    [0.03, 0.03, 0.06],
    "bread":   [0.05, 0.03, 0.03],
    "cereal":  [0.04, 0.02, 0.07],
    "lemon":   [0.025, 0.025, 0.025],
}


def _make_material(colour: str, idx: int) -> CustomMaterial:
    """Create a robosuite ``CustomMaterial`` for *colour*."""
    _, tex = COLOUR_PALETTE.get(colour, ([0.5, 0.5, 0.5, 1], "WoodLight"))
    return CustomMaterial(
        texture=tex,
        tex_name=f"{colour}wood_{idx}",
        mat_name=f"{colour}wood_mat_{idx}",
        tex_attrib={"type": "cube"},
        mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
    )


def _create_object(
    spec: dict[str, Any], idx: int
) -> tuple[MujocoObject, np.ndarray]:
    """Instantiate a robosuite object from a spec dict.

    Returns ``(obj, extent)`` where *extent* is the full bounding-box size
    ``[x, y, z]`` in metres.
    """
    obj_type = spec.get("type", "box")
    name = spec["name"]
    colour = spec.get("colour")
    size = spec.get("size")

    rgba = None
    material = None
    if colour and obj_type in ("box", "ball", "cylinder", "cone"):
        rgba = COLOUR_PALETTE.get(colour, ([0.5, 0.5, 0.5, 1], "WoodLight"))[0]
        material = _make_material(colour, idx)

    if obj_type in ("box", "cube"):
        sz = size or _DEFAULT_SIZE["box"]
        obj = BoxObject(
            name=name,
            size_min=sz, size_max=sz,
            rgba=rgba, material=material,
        )
        extent = np.array(sz) * 2

    elif obj_type in ("ball", "sphere"):
        sz = size or _DEFAULT_SIZE["ball"]
        obj = BallObject(
            name=name,
            size_min=sz, size_max=sz,
            rgba=rgba, material=material,
        )
        r = sz[0]
        extent = np.array([r * 2, r * 2, r * 2])

    elif obj_type == "cylinder":
        sz = size or _DEFAULT_SIZE["cylinder"]
        obj = CylinderObject(
            name=name,
            size_min=sz, size_max=sz,
            rgba=rgba, material=material,
        )
        r, h = sz[0], sz[1]
        extent = np.array([r * 2, r * 2, h * 2])

    elif obj_type == "cone":
        outer_r = 0.0425
        height = 0.05
        obj = ConeObject(name=name, outer_radius=outer_r, height=height, rgba=rgba)
        extent = np.array([outer_r * 2, outer_r * 2, height])

    elif obj_type == "bottle":
        obj = BottleObject(name=name)
        extent = np.array(_XML_BBOX_HALF["bottle"]) * 2

    elif obj_type == "can":
        obj = CanObject(name=name)
        extent = np.array(_XML_BBOX_HALF["can"]) * 2

    elif obj_type == "milk":
        obj = MilkObject(name=name)
        extent = np.array(_XML_BBOX_HALF["milk"]) * 2

    elif obj_type == "bread":
        obj = BreadObject(name=name)
        extent = np.array(_XML_BBOX_HALF["bread"]) * 2

    elif obj_type == "cereal":
        obj = CerealObject(name=name)
        extent = np.array(_XML_BBOX_HALF["cereal"]) * 2

    elif obj_type == "lemon":
        obj = LemonObject(name=name)
        extent = np.array(_XML_BBOX_HALF["lemon"]) * 2

    else:
        sz = size or _DEFAULT_SIZE["box"]
        obj = BoxObject(
            name=name,
            size_min=sz, size_max=sz,
            rgba=rgba, material=material,
        )
        extent = np.array(sz) * 2

    return obj, extent


# ── Robosuite environment ──────────────────────────────────────────────────

class MultiObjectTable(ManipulationEnv):
    """Robosuite table environment with configurable objects.

    Parameters
    ----------
    object_specs : list[dict]
        Each dict has keys ``name`` (str), ``type`` (str), and optionally
        ``colour`` (str) and ``size`` (list[float]).
    """

    def __init__(
        self,
        robots,
        object_specs: list[dict[str, Any]] | None = None,
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        placement_initializer=None,
        **kwargs,
    ):
        self.object_specs = object_specs or [
            {"name": "red_cube_1", "type": "box", "colour": "red"},
            {"name": "green_cube_1", "type": "box", "colour": "green"},
        ]
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))
        self.use_object_obs = True
        self.reward_scale = 1.0
        self.reward_shaping = kwargs.pop("reward_shaping", True)
        self.placement_initializer = placement_initializer

        self.objects: dict[str, MujocoObject] = {}
        self.object_body_ids: dict[str, int] = {}
        self.object_extents: dict[str, np.ndarray] = {}

        super().__init__(
            robots=robots,
            use_camera_obs=kwargs.pop("use_camera_obs", False),
            has_renderer=kwargs.pop("has_renderer", False),
            has_offscreen_renderer=kwargs.pop("has_offscreen_renderer", True),
            **kwargs,
        )

    def _load_model(self):
        super()._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        obj_list: list[MujocoObject] = []
        for idx, spec in enumerate(self.object_specs):
            obj, extent = _create_object(spec, idx)
            self.objects[spec["name"]] = obj
            self.object_extents[spec["name"]] = extent
            obj_list.append(obj)

        n = len(obj_list)
        x_range = min(0.35, 0.04 * n + 0.08)
        y_range = min(0.28, 0.03 * n + 0.06)

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(obj_list)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=obj_list,
                x_range=[-x_range, x_range],
                y_range=[-y_range, y_range],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                rng=self.rng,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=obj_list,
        )

    def _setup_references(self):
        super()._setup_references()
        for name, obj in self.objects.items():
            self.object_body_ids[name] = self.sim.model.body_name2id(obj.root_body)

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                )

    def _setup_observables(self):
        observables = super()._setup_observables()
        if not self.use_object_obs:
            return observables

        modality = "object"
        for name in list(self.objects.keys()):
            bid = self.object_body_ids[name]

            def _make_pos_sensor(b=bid, n=name):
                @sensor(modality=modality)
                def pos(obs_cache):
                    return np.array(self.sim.data.body_xpos[b])
                pos.__name__ = f"{n}_pos"
                return pos

            def _make_quat_sensor(b=bid, n=name):
                @sensor(modality=modality)
                def quat(obs_cache):
                    return convert_quat(np.array(self.sim.data.body_xquat[b]), to="xyzw")
                quat.__name__ = f"{n}_quat"
                return quat

            for s in [_make_pos_sensor(), _make_quat_sensor()]:
                observables[s.__name__] = Observable(
                    name=s.__name__, sensor=s, sampling_rate=self.control_freq
                )

        return observables

    def reward(self, action=None):
        return 0.0

    def staged_rewards(self):
        return 0.0, 0.0, 0.0

    def _check_success(self):
        return False


# ── CaP-X low-level wrapper ────────────────────────────────────────────────

class FrankaRobosuiteMultiObjectsLowLevel(RobosuiteBaseEnv):
    """CaP-X wrapper around ``MultiObjectTable``."""

    _SUBSAMPLE_RATE = 5

    def __init__(
        self,
        object_specs: list[dict[str, Any]] | None = None,
        cube_specs: list[tuple[str, str]] | None = None,
        controller_cfg: str = "capx/integrations/robosuite/controllers/config/robots/panda_joint_ctrl.json",
        max_steps: int | None = None,
        seed: int | None = None,
        viser_debug: bool = False,
        privileged: bool = False,
        enable_render: bool = False,
    ) -> None:
        # Backward compat: convert old cube_specs to object_specs
        if object_specs is None and cube_specs is not None:
            object_specs = [
                {"name": name, "type": "box", "colour": colour}
                for name, colour in cube_specs
            ]
        if object_specs is None:
            object_specs = [
                {"name": "red_cube_1", "type": "box", "colour": "red"},
                {"name": "green_cube_1", "type": "box", "colour": "green"},
            ]

        if max_steps is None:
            max_steps = max(3000, 500 * len(object_specs))

        super().__init__(
            controller_cfg=controller_cfg,
            max_steps=max_steps,
            seed=seed,
            viser_debug=False,
            privileged=privileged,
            enable_render=enable_render,
        )
        self.object_specs = object_specs

        common_kwargs = dict(
            robots=["Panda"],
            object_specs=self.object_specs,
            camera_names=self.render_camera_names,
            renderer="mujoco",
            camera_heights=self._render_height,
            camera_widths=self._render_width,
            controller_configs=load_composite_controller_config(controller=self.controller_cfg),
            horizon=max_steps,
        )

        if privileged:
            if not enable_render:
                self.render_camera_names = []
                self.robosuite_env = MultiObjectTable(
                    use_camera_obs=False,
                    has_renderer=False,
                    has_offscreen_renderer=False,
                    camera_names=self.render_camera_names,
                    **{k: v for k, v in common_kwargs.items() if k != "camera_names"},
                )
            else:
                self.robosuite_env = MultiObjectTable(
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    camera_depths=True,
                    **common_kwargs,
                )
        else:
            self.robosuite_env = MultiObjectTable(
                has_renderer=True,
                has_offscreen_renderer=True,
                camera_depths=True,
                **common_kwargs,
            )

        self._init_robot_links()
        self._init_viser_debug(viser_debug)

    @property
    def object_extents(self) -> dict[str, np.ndarray]:
        """Per-object full bounding-box extents ``{name: [x, y, z]}``."""
        return self.robosuite_env.object_extents

    # ── Reset ───────────────────────────────────────────────────────────────

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.robosuite_env.reset()
        self.robosuite_env.sim.data.qpos[6] -= np.pi

        self._step_count = 0
        self._sim_step_count = 0

        for _ in range(50):
            self.robosuite_env.sim.forward()
            self.robosuite_env.sim.step()
            self._set_gripper(1.0)

        robosuite_obs = self.robosuite_env._get_observations()
        self._current_joints = np.array(robosuite_obs["robot0_joint_pos"], dtype=np.float64)
        self._current_joints[6] -= np.pi

        obs = self.get_observation()
        self.gripper_link_wxyz_xyz = np.concatenate([
            self.robosuite_env.sim.data.xquat[self.gripper_link_idx],
            self.robosuite_env.sim.data.xpos[self.gripper_link_idx],
        ])

        obj_names = [s["name"] for s in self.object_specs]
        info = {
            "task_prompt": f"Manipulate objects on the table. Available objects: {obj_names}. Quaternions are WXYZ.",
        }
        return obs, info

    # ── Observation ─────────────────────────────────────────────────────────

    def _object_pose_dict(self, robosuite_obs: dict[str, Any]) -> dict[str, list[float]]:
        """Get all object poses in robot base frame."""
        base_link_wxyz_xyz = np.concatenate([
            self.robosuite_env.sim.data.xquat[self.base_link_idx],
            self.robosuite_env.sim.data.xpos[self.base_link_idx],
        ])
        base_transform = vtf.SE3(wxyz_xyz=base_link_wxyz_xyz).inverse()

        result: dict[str, list[float]] = {}
        for name in self.robosuite_env.object_body_ids:
            pos_key = f"{name}_pos"
            quat_key = f"{name}_quat"
            if pos_key in robosuite_obs and quat_key in robosuite_obs:
                obj_world = vtf.SE3(
                    wxyz_xyz=np.concatenate([
                        convert_quat(robosuite_obs[quat_key], to="wxyz"),
                        robosuite_obs[pos_key],
                    ])
                )
            else:
                bid = self.robosuite_env.object_body_ids[name]
                obj_world = vtf.SE3(
                    wxyz_xyz=np.concatenate([
                        self.robosuite_env.sim.data.body_xquat[bid],
                        self.robosuite_env.sim.data.body_xpos[bid],
                    ])
                )
            obj_robot_base = base_transform @ obj_world
            result[name] = [
                float(x)
                for x in np.concatenate([obj_robot_base.translation(), obj_robot_base.rotation().wxyz])
            ]
        return result

    def compute_reward(self) -> float:
        return 0.0

    def task_completed(self) -> bool | None:
        return None

    def get_observation(self) -> dict[str, Any]:
        robosuite_obs = self.robosuite_env._get_observations()
        pose_dict = self._object_pose_dict(robosuite_obs)
        robosuite_obs["object_poses"] = pose_dict
        robosuite_obs["cube_poses"] = pose_dict  # backward compat
        self._process_camera_observations(robosuite_obs)
        self._compute_gripper_obs(robosuite_obs)
        return robosuite_obs


# Backward-compat aliases
MultiCubeStack = MultiObjectTable
FrankaRobosuiteMultiCubesLowLevel = FrankaRobosuiteMultiObjectsLowLevel

__all__ = [
    "MultiObjectTable",
    "FrankaRobosuiteMultiObjectsLowLevel",
    "MultiCubeStack",
    "FrankaRobosuiteMultiCubesLowLevel",
    "COLOUR_PALETTE",
]
