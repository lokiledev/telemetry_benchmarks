from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class NamedTransform:
    parent: str
    child: str
    mat: NDArray[np.float64]  # 4x4 transform matrix


class Logger(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def log_joint_states(self, qpos: np.ndarray) -> None:
        pass

    @abstractmethod
    def log_video(self, video: np.ndarray) -> None:
        pass

    @abstractmethod
    def log_end_effector_pose(self, pose: np.ndarray) -> None:
        pass

    @abstractmethod
    def log_transforms(self, transforms: list[NamedTransform]) -> None:
        pass


class NullLogger(Logger):
    def log_joint_states(self, qpos: np.ndarray) -> None:
        pass

    def log_video(self, video: np.ndarray) -> None:
        pass

    def log_end_effector_pose(self, pose: np.ndarray) -> None:
        pass

    def log_transforms(self, transforms: list[NamedTransform]) -> None:
        pass
