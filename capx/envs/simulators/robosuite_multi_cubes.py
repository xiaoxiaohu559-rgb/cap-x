"""Backward-compatibility shim — re-exports from ``robosuite_multi_objects``."""

from capx.envs.simulators.robosuite_multi_objects import (  # noqa: F401
    COLOUR_PALETTE,
    FrankaRobosuiteMultiCubesLowLevel,
    FrankaRobosuiteMultiObjectsLowLevel,
    MultiCubeStack,
    MultiObjectTable,
)

FrankaRobosuiteMultiCubesLowLevel = FrankaRobosuiteMultiCubesLowLevel
MultiCubeStack = MultiCubeStack

__all__ = ["MultiCubeStack", "FrankaRobosuiteMultiCubesLowLevel", "COLOUR_PALETTE"]
