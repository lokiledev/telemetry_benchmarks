from pathlib import Path
from typing import Literal

import genesis as gs
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from telemetry_benchmarks.sim.config import CAMERA_FPS, CAMERA_RESOLUTION, OUTPUT_DIR
from telemetry_benchmarks.sim.datalogger import DataLogger, NamedTransform, NullLogger
from telemetry_benchmarks.sim.mcap_datalogger import MCAPLogger

GraspState = Literal["idle", "grasp", "lift", "end"]
ASSET_DIR = Path(__file__).parent / "SO101"
ROBOT_MJCF = ASSET_DIR / "so101_new_calib.xml"

CUBE_INITIAL_POSE = np.array([0.25, 0.0, 0.02])
EEF_TARGET_POSE = CUBE_INITIAL_POSE + np.array([0.0, 0.0, 0.01])
EEF_TARGET_QUAT = np.array([1, 0, 0, 0])  # wxyz


def quat_to_rot_mat(quat: NDArray[np.float64]) -> NDArray[np.float64]:
    return Rotation.from_quat(quat).as_matrix()


def pose_to_transform(
    translation: NDArray[np.float64], quaternion: NDArray[np.float64]
) -> NDArray[np.float64]:
    rot_mat = quat_to_rot_mat(quaternion)
    transform = np.eye(4)
    transform[:3, :3] = rot_mat
    transform[:3, 3] = translation
    return transform


class Env:
    def __init__(self, logger: DataLogger | None = None):
        self.logger = logger or NullLogger()
        self.last_camera_timestamp = 0
        self.step_number = 0
        self.phase: GraspState = "idle"
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0.0, 0.0, 0.25),
                camera_fov=30,
                res=(960, 640),
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(
                dt=0.004,  # 250 Hz x4 = 1KHz
                substeps=4,
            ),
            rigid_options=gs.options.RigidOptions(
                box_box_detection=True,
            ),
            show_viewer=True,
        )
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file=str(ROBOT_MJCF)),
        )
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(0.03, 0.03, 0.03), pos=CUBE_INITIAL_POSE),
        )
        self.camera = self.scene.add_camera(
            res=CAMERA_RESOLUTION,
            pos=(3, -1, 1.5),
            lookat=(0.0, 0.0, 0.2),
            fov=30,
            GUI=True,
        )

        self.scene.build()

        joints = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
        self.dofs_idx = [self.robot.get_joint(name).dof_idx_local for name in joints]
        self.motors_dof = np.array(self.dofs_idx[:-1])
        self.gripper_dof = np.array(self.dofs_idx[-1])
        self.robot.set_dofs_kp(
            np.array([100.0] * len(self.dofs_idx)),
            dofs_idx_local=self.dofs_idx,
        )
        self.robot.set_dofs_kv(
            np.array([5] * len(self.dofs_idx)),
            dofs_idx_local=self.dofs_idx,
        )
        self.qpos = np.array([0.0, 0.0, 0.0, 1.5708, 0.0, 1.5])
        self.robot.set_qpos(self.qpos)

        self.end_effector = self.robot.get_link("gripper")
        cam_offset = np.array([0.1, 0.0, 0.1])
        rot_offset_mat = np.array(
            [
                0.9396926,
                0.0000000,
                0.3420202,
                0.0000000,
                1.0000000,
                0.0000000,
                -0.3420202,
                0.0000000,
                0.9396926,
            ]
        ).reshape(3, 3)
        cam_offset_mat = np.eye(4)
        cam_offset_mat[:3, :3] = rot_offset_mat
        cam_offset_mat[:3, 3] = cam_offset
        self.camera.attach(self.end_effector, cam_offset_mat)
        self.qpos = self.robot.inverse_kinematics(
            link=self.end_effector,
            pos=EEF_TARGET_POSE,
            quat=EEF_TARGET_QUAT,
        )
        self.robot.control_dofs_position(self.qpos, self.dofs_idx)
        self.scene.step()

    def state_from_timestamp(self, timestamp: float) -> GraspState:
        if timestamp <= 0.1:
            return "idle"
        elif timestamp <= 0.6:
            return "grasp"
        elif timestamp <= 1.5:
            return "lift"
        else:
            return "end"

    def observe(self, timestamp: float) -> None:
        self.camera.move_to_attach()
        if timestamp == 0 or (
            timestamp - self.last_camera_timestamp > 1.0 / CAMERA_FPS
        ):
            color, _, _, _ = self.camera.render()
            self.logger.log_video(color, timestamp)
            self.last_camera_timestamp = timestamp
            transforms = self.get_link_transforms()
            self.logger.log_transforms(transforms, timestamp)
        qpos = self.robot.get_qpos(self.dofs_idx).cpu().numpy()
        self.logger.log_joint_states(qpos, timestamp)

    def get_link_transforms(self) -> list[NamedTransform]:
        geoms = self.robot.geoms
        geoms_T = self.scene.rigid_solver._geoms_render_T
        transforms = []
        seen = set()
        for geom in geoms:
            tf = geoms_T[geom.idx, 0]
            link_name = geom.link.name
            if link_name in seen:
                continue
            seen.add(link_name)
            named_tf = NamedTransform(parent="world", child=link_name, mat=tf)
            transforms.append(named_tf)
        return transforms

    def act(self, timestamp: float) -> None:
        self.phase = self.state_from_timestamp(timestamp)
        if self.phase == "idle":
            pass
        if self.phase == "grasp":
            self.gripper_pos = -0.05
            self.qpos[self.gripper_dof] = self.gripper_pos
            # grasp
            self.robot.control_dofs_position(self.qpos, self.dofs_idx)
        elif self.phase == "lift":
            target_pose = EEF_TARGET_POSE + np.array([-0.1, 0.0, 0.2])
            self.scene.draw_debug_sphere(
                target_pose, radius=0.01, color=(0.0, 1.0, 0.0, 0.5)
            )
            self.qpos = self.robot.inverse_kinematics(
                link=self.end_effector,
                pos=target_pose,
                quat=EEF_TARGET_QUAT,
            )
            self.qpos[self.gripper_dof] = self.gripper_pos
            self.robot.control_dofs_position(self.qpos, self.dofs_idx)

    def run(self) -> None:
        while self.phase != "end":
            timestamp = self.step_number * self.scene.dt
            self.observe(timestamp)
            self.act(timestamp)
            self.scene.step()
            self.step_number += 1
        self.logger.finish()


def main():
    ########################## init ##########################
    gs.init(backend=gs.cpu, precision="32")
    env = Env(logger=MCAPLogger(OUTPUT_DIR / "robot_arm.mcap"))
    ############## run the environment #####################
    env.run()


if __name__ == "__main__":
    main()
