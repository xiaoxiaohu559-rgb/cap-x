"""Code-execution task environment for multi-cube manipulation.

The prompt is set dynamically (via ``cfg.prompt``) based on the actual cubes
present in the simulation, so the class itself has ``prompt = None``.
"""

from capx.envs.tasks.base import CodeExecutionEnvBase


class FrankaMultiCubeCodeEnv(CodeExecutionEnvBase):
    """High-level code environment for Franka multi-cube manipulation."""

    prompt = None


__all__ = ["FrankaMultiCubeCodeEnv"]
