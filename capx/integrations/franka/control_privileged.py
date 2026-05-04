from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

from capx.envs.base import (
    BaseEnv,
)
from capx.integrations.base_api import ApiBase
from capx.integrations.franka.common import (
    apply_tcp_offset,
    close_gripper as _close_gripper,
    open_gripper as _open_gripper,
)
from capx.integrations.motion.pyroki import init_pyroki


# ------------------------------- Control API ------------------------------
class FrankaControlPrivilegedApi(ApiBase):
    """Robot control helpers for Franka.

    Functions:
      - get_object_pose(object_name: str) -> (position: np.ndarray, quaternion_wxyz: np.ndarray):
      - sample_grasp_pose(object_name: str) -> (position: np.ndarray, quaternion_wxyz: np.ndarray):
      - goto_pose(position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0) -> None
      - open_gripper() -> None
      - close_gripper() -> None
      - pick_object(object_name: str, max_attempts: int = 3) -> bool
      - is_grasping(object_name: str) -> bool
    """

    _TCP_OFFSET = np.array([0.0, 0.0, -0.107], dtype=np.float64)

    def __init__(self, env: BaseEnv, multi_turn: bool = False) -> None:
        super().__init__(env)
        # Lazy-import to keep startup light
        # from capx.integrations.motion import pyroki_snippets as pks  # type: ignore
        # from capx.integrations.motion.pyroki_context import get_pyroki_context  # type: ignore

        # ctx = get_pyroki_context("panda_description", target_link_name="panda_hand")
        # self._robot = ctx.robot
        # self._target_link_name = ctx.target_link_name
        # self._pks = pks
        self.ik_solve_fn = init_pyroki()
        self.cfg = None
        self.multi_turn = multi_turn

    def functions(self) -> dict[str, Any]:
        base_functions = {
            "get_object_pose": self.get_object_pose,
            "sample_grasp_pose": self.sample_grasp_pose,
            "goto_pose": self.goto_pose,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            "pick_object": self.pick_object,
            "is_grasping": self.is_grasping,
            "get_grasp_history": self.get_grasp_history,
        }
        return base_functions

    @staticmethod
    def _match_object(object_name: str, poses: dict) -> tuple[str, np.ndarray] | tuple[None, None]:
        """Match a natural-language object name to a key in *poses*.

        Returns ``(matched_key, pose_array)`` or ``(None, None)``.
        """
        query = object_name.lower().strip()

        # Legacy 2-cube layout
        if "primary" in poses:
            if "red" in query and "cube" in query:
                return "primary", np.asarray(poses["primary"], dtype=np.float64)
            if "green" in query and "cube" in query:
                return "secondary", np.asarray(poses["secondary"], dtype=np.float64)
            return None, None

        # Exact match after normalising spaces → underscores
        normalised = query.replace(" ", "_")
        if normalised in poses:
            return normalised, np.asarray(poses[normalised], dtype=np.float64)

        # Fuzzy: check if all key tokens appear in the query
        for key, pose in poses.items():
            key_parts = key.lower().split("_")
            if all(part in query for part in key_parts):
                return key, np.asarray(pose, dtype=np.float64)

        return None, None

    def get_object_pose(
        self, object_name: str, return_bbox_extent: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Get the pose of an object in the environment from a natural language description.
        The quaternion from get_object_pose may be unreliable, so disregard it and use the grasp pose quaternion OR (0, 0, 1, 0) wxyz as the gripper down orientation if using this for placement position.

        Args:
            object_name: The name of the object to get the pose of.

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
            bbox_extent: (3,) object extent in meters of x, y, z axes respectively in the world frame (full side length, not half-length extent). If return_bbox_extent is False, returns None.
        """
        obs = self._env.get_observation()
        poses = obs.get("object_poses") or obs.get("cube_poses", {})
        matched_key, pose = self._match_object(object_name, poses)
        if pose is not None:
            extent = np.array([0.05, 0.05, 0.05])
            env_extents = getattr(self._env, "object_extents", None)
            if env_extents and matched_key in env_extents:
                extent = np.asarray(env_extents[matched_key], dtype=np.float64)
            return (pose[:3], pose[3:], extent)
        available = list(poses.keys())
        raise ValueError(f"Invalid object name: {object_name}. Available objects: {available}")

    def sample_grasp_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Sample a grasp pose for an object in the environment from a natural language description.
        Do use the grasp sample quaternion from sample_grasp_pose.

        Args:
            object_name: The name of the object to sample a grasp pose for.

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        """
        obs = self._env.get_observation()
        poses = obs.get("object_poses") or obs.get("cube_poses", {})
        _, pose = self._match_object(object_name, poses)
        if pose is not None:
            return pose[:3], np.array([0, 0, 1, 0])
        available = list(poses.keys())
        raise ValueError(f"Invalid object name: {object_name}. Available objects: {available}")

    def goto_pose(
        self, position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0
    ) -> None:
        """Go to pose using Inverse Kinematics.
        There is no need to call a second goto_pose with the same position and quaternion_wxyz after calling it with z_approach.
        Args:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
            z_approach: (float) Z-axis distance offset for goto_pose insertion approach motion. Will first arrive at position + z_approach meters in Z-axis before moving to the requested pose. Useful for more precise grasp approaches. Default is 0.0.
        Returns:
            None
        """

        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        offset_pos = apply_tcp_offset(pos, quat_wxyz, self._TCP_OFFSET)
        rot = SciRotation.from_quat(
            np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
        )

        if z_approach != 0.0:
            z_offset_pos = offset_pos + rot.apply(np.array([0, 0, -z_approach]))
            self._solve_and_move(quat_wxyz, z_offset_pos)

        self._solve_and_move(quat_wxyz, offset_pos)

    def _solve_and_move(self, quat_wxyz: np.ndarray, target_pos: np.ndarray) -> None:
        """Solve IK and move to target position (helper to reduce goto_pose duplication)."""
        if self.cfg is None:
            self.cfg = self.ik_solve_fn(
                target_pose_wxyz_xyz=np.concatenate([quat_wxyz, target_pos]),
            )
        else:
            self.cfg = self.ik_solve_fn(
                target_pose_wxyz_xyz=np.concatenate([quat_wxyz, target_pos]),
                prev_cfg=self.cfg,
            )
        joints = np.asarray(self.cfg[:-1], dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking(joints)

    def open_gripper(self) -> None:
        """Open gripper fully.

        Args:
            None
        """
        _open_gripper(self._env, steps=20)

    def close_gripper(self) -> None:
        """Close gripper fully.

        Args:
            None
        """
        _close_gripper(self._env, steps=30)

    def is_grasping(self, object_name: str) -> bool:
        """Check if the named object is currently grasped (lifted with the gripper).
        Call after close_gripper() and lifting to verify grasp success.

        Args:
            object_name: The name of the object to check.

        Returns:
            True if the object appears to be held by the gripper.
        """
        obs = self._env.get_observation()
        poses = obs.get("object_poses") or obs.get("cube_poses", {})
        _, pose = self._match_object(object_name, poses)
        if pose is None:
            return False
        obj_z = pose[2]
        gripper_pos = obs.get("robot_cartesian_pos")
        if gripper_pos is None:
            return False
        gripper_z = float(gripper_pos[2])
        return abs(obj_z - gripper_z) < 0.08

    def pick_object(self, object_name: str, max_attempts: int = 3) -> bool:
        """Pick up an object with automatic retry on failure.
        On retry, applies a small random XY offset to the grasp position.
        Returns True if grasp succeeded, False if all attempts failed.

        Args:
            object_name: The name of the object to pick up.
            max_attempts: Maximum number of grasp attempts.

        Returns:
            True if the object was successfully grasped.
        """
        rng = np.random.default_rng()
        for attempt in range(1, max_attempts + 1):
            pos, quat = self.sample_grasp_pose(object_name)
            if attempt > 1:
                offset = rng.uniform(-0.01, 0.01, size=2)
                pos[0] += offset[0]
                pos[1] += offset[1]
            self.open_gripper()
            self.goto_pose(pos, quat, z_approach=0.1)
            self.close_gripper()
            lift = pos.copy()
            lift[2] += 0.15
            self.goto_pose(lift, quat)

            success = self.is_grasping(object_name)
            self._log_grasp_attempt(object_name, pos, quat, success, attempt)

            if success:
                return True
            self.open_gripper()
        return False

    _GRASP_LOG_PATH = Path("outputs/grasp_log.jsonl")

    def _log_grasp_attempt(
        self,
        object_name: str,
        grasp_pos: np.ndarray,
        grasp_quat: np.ndarray,
        success: bool,
        attempt: int,
    ) -> None:
        """Append a grasp attempt record to the log file."""
        obs = self._env.get_observation()
        poses = obs.get("object_poses") or obs.get("cube_poses", {})
        _, pose = self._match_object(object_name, poses)
        gripper_pos = obs.get("robot_cartesian_pos")

        obj_type = "unknown"
        env_extents = getattr(self._env, "object_specs", None)
        if env_extents:
            for spec in env_extents:
                if spec.get("name") == object_name:
                    obj_type = spec.get("type", "unknown")
                    break

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "object_name": object_name,
            "object_type": obj_type,
            "object_pos": [float(x) for x in pose[:3]] if pose is not None else None,
            "grasp_pos": [float(x) for x in grasp_pos],
            "grasp_quat": [float(x) for x in grasp_quat],
            "success": bool(success),
            "attempt": int(attempt),
            "gripper_z_after_lift": float(gripper_pos[2]) if gripper_pos is not None else None,
            "object_z_after_lift": float(pose[2]) if pose is not None else None,
        }
        self._GRASP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(self._GRASP_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_grasp_history(self, object_name: str | None = None) -> list[dict]:
        """Return past grasp attempts from the log file.

        Args:
            object_name: If given, only return attempts for this object.
                         If None, return all attempts.

        Returns:
            List of dicts with keys: object_name, object_type, grasp_pos,
            success, attempt, etc.
        """
        if not self._GRASP_LOG_PATH.exists():
            return []
        results: list[dict] = []
        with open(self._GRASP_LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if object_name is None or entry.get("object_name") == object_name:
                    results.append(entry)
        return results

    def home_pose(self) -> None:
        """
        Move the robot to a safe home pose.
        Args:
            None
        Returns:
            None
        """

        # joints = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])
        joints = np.array(
            [
                -2.95353726e-02,
                1.69197371e-01,
                2.39244731e-03,
                -2.64089311e00,
                -2.01237851e-03,
                2.94565778e00,
                8.31390616e-01,
            ]
        )
        self._env.move_to_joints_blocking(joints)
