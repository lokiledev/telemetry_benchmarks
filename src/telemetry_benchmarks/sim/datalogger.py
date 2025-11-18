from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class NamedTransform:
    parent: str
    child: str
    mat: NDArray[np.float64]  # 4x4 transform matrix


class DataLogger(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def log_joint_states(self, qpos: np.ndarray, timestamp: float) -> None:
        pass

    @abstractmethod
    def log_video(self, video: np.ndarray, timestamp: float) -> None:
        pass

    @abstractmethod
    def log_end_effector_pose(self, pose: np.ndarray, timestamp: float) -> None:
        pass

    @abstractmethod
    def log_transforms(
        self, transforms: list[NamedTransform], timestamp: float
    ) -> None:
        pass

    @abstractmethod
    def finish(self) -> None:
        pass


class NullLogger(DataLogger):
    def log_joint_states(self, qpos: np.ndarray, timestamp: float) -> None:
        pass

    def log_video(self, video: np.ndarray, timestamp: float) -> None:
        pass

    def log_end_effector_pose(self, pose: np.ndarray, timestamp: float) -> None:
        pass

    def log_transforms(
        self, transforms: list[NamedTransform], timestamp: float
    ) -> None:
        pass

    def finish(self) -> None:
        pass
