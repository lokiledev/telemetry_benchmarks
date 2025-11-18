from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


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
    def log_transforms(self, transforms: list[Tuple[str, np.ndarray]]) -> None:
        pass


class NullLogger(Logger):
    def log_joint_states(self, qpos: np.ndarray) -> None:
        pass

    def log_video(self, video: np.ndarray) -> None:
        pass

    def log_end_effector_pose(self, pose: np.ndarray) -> None:
        pass

    def log_transforms(self, transforms: list[Tuple[str, np.ndarray]]) -> None:
        pass
