from pathlib import Path
from typing import Literal

import cyclopts
import genesis as gs
import numpy as np
from genesis.utils.geom import trans_quat_to_T

from telemetry_benchmarks.sim.config import CAMERA_FPS, CAMERA_RESOLUTION, OUTPUT_DIR
from telemetry_benchmarks.sim.datalogger import DataLogger, NamedTransform, NullLogger
from telemetry_benchmarks.sim.mcap_datalogger import MCAPLogger
from telemetry_benchmarks.sim.rerun_datalogger import RerunLogger

GraspState = Literal["idle", "grasp", "lift", "end"]
ASSET_DIR = Path(__file__).parent / "SO101"
ROBOT_URDF = ASSET_DIR / "so101_new_calib.urdf"

CUBE_INITIAL_POSE = np.array([0.25, 0.0, 0.02])
EEF_TARGET_POSE = CUBE_INITIAL_POSE + np.array([-0.02, 0.0, -0.01])
EEF_TARGET_RPY = np.array([0.0, 0.0, 0.0])
EEF_TARGET_QUAT = np.array([0.0, 0.0, 1.0, 0.0])


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
                camera_fov=30,  # RealSense D455 horizontal FOV
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
            show_FPS=False,
        )
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(file=str(ROBOT_URDF), merge_fixed_links=False, fixed=True),
        )
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(0.03, 0.03, 0.03), pos=CUBE_INITIAL_POSE),
        )
        self.camera = self.scene.add_camera(
            res=CAMERA_RESOLUTION,
            pos=(3, -1, 1.5),
            lookat=CUBE_INITIAL_POSE,
            fov=87,  # RealSense D455 horizontal FOV
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
        self.qpos = np.array([0.0, 0.0, 0.0, 1.57, 0.0, 1.5])
        self.robot.set_qpos(self.qpos)
        self.end_effector = self.robot.get_link("gripper_frame_link")
        cam_offset = np.array([0.1, 0.0, 0.2])
        cam_offset_mat = np.eye(4)
        cam_offset_mat[:3, 3] = cam_offset
        gripper_base = self.robot.get_link("gripper_link")
        self.camera.attach(gripper_base, cam_offset_mat)
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
        transforms = []
        for link in self.robot.links:
            if link.name == "base_link":
                continue
            tf = trans_quat_to_T(link.get_pos(), link.get_quat())
            named_tf = NamedTransform(parent="base_link", child=link.name, mat=tf)
            transforms.append(named_tf)
        cam_named_tf = NamedTransform(
            parent="base_link", child="camera_link", mat=self.camera.transform
        )
        transforms.append(cam_named_tf)
        return transforms

    def act(self, timestamp: float) -> None:
        self.phase = self.state_from_timestamp(timestamp)
        if self.phase == "idle":
            pass
        if self.phase == "grasp":
            self.gripper_pos = 0.02
            self.qpos[self.gripper_dof] = self.gripper_pos
            # grasp
            self.robot.control_dofs_position(self.qpos, self.dofs_idx)
        elif self.phase == "lift":
            target_pose = EEF_TARGET_POSE + np.array([0.0, 0.0, 0.2])
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


def main(logger_type: Literal["mcap", "rerun"] = "mcap"):
    ########################## init ##########################
    match logger_type:
        case "mcap":
            data_logger = MCAPLogger(OUTPUT_DIR / "robot_arm.mcap")
        case "rerun":
            data_logger = RerunLogger(OUTPUT_DIR / "robot_arm.rrd", ROBOT_URDF)
    gs.init(backend=gs.cpu, precision="32")
    env = Env(logger=data_logger)
    ############## run the environment #####################
    env.run()


if __name__ == "__main__":
    cyclopts.run(main)
