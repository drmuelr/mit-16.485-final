# Bandaid for circular import. Real solution is to not have circular depdenence.
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from traj_opt.optimizer.optimizer import Optimizer

from abc import ABC, abstractmethod

from casadi import MX, SX

from traj_opt.models.terrain.base import TerrainBase


__all__ = [
    "RobotBase",
]


class RobotBase(ABC):

    @abstractmethod
    def __init__(self, optimizer: Optimizer, terrain: TerrainBase):
        self.position_world: list[MX]
        self.velocity_world: list[MX]
        self.q_body_to_world: list[MX]
        self.angular_velocity_body: list[MX]
        self.control_thrusts: list[MX]
        self.control_moment_body: list[MX]
