from typing import Literal

import genesis as gs
import numpy as np

from telemetry_benchmarks.sim.datalogger import Logger, NamedTransform, NullLogger

GraspState = Literal["idle", "grasp", "lift", "end"]


class Env:
    def __init__(self, logger: Logger | None = None):
        self.logger = logger or NullLogger()
        self.last_camera_timestamp = 0
        self.step_number = 0
        self.phase: GraspState = "idle"
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
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
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        )
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.02)),
        )
        self.camera = self.scene.add_camera(
            res=(640, 480), pos=(3, -1, 1.5), lookat=(0.0, 0.0, 0.5), fov=30
        )

        self.scene.build()

        self.motors_dof = np.array(np.arange(7))
        self.fingers_dof = np.array(np.arange(7, 9))
        self.qpos = np.array(
            [-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04]
        )
        self.franka.set_qpos(self.qpos)
        self.scene.step()

        self.end_effector = self.franka.get_link("hand")
        self.qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=np.array([0.65, 0.0, 0.135]),
            quat=np.array([0, 1, 0, 0]),
        )
        self.franka.control_dofs_position(self.qpos[:-2], self.motors_dof)
        self.finger_pos = 0.5

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
        if timestamp - self.last_camera_timestamp >= 1.0 / 30:
            color, _, _, _ = self.camera.render()
            self.logger.log_video(color)
            self.last_camera_timestamp = timestamp
        qpos = (
            self.franka.get_qpos(np.concatenate([self.motors_dof, self.fingers_dof]))
            .cpu()
            .numpy()
        )
        self.logger.log_joint_states(qpos)

        transforms = self.get_link_transforms()
        self.logger.log_transforms(transforms)

    def get_link_transforms(self) -> list[NamedTransform]:
        geoms = self.franka.geoms
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
        finger_pos = -0.0
        self.phase = self.state_from_timestamp(timestamp)
        if self.phase == "idle":
            pass
        if self.phase == "grasp":
            self.finger_pos = -0.0
            # grasp
            self.franka.control_dofs_position(self.qpos[:-2], self.motors_dof)
            self.franka.control_dofs_position(
                np.array([finger_pos, finger_pos]), self.fingers_dof
            )
        elif self.phase == "lift":
            self.qpos = self.franka.inverse_kinematics(
                link=self.end_effector,
                pos=np.array([0.65, 0.0, 0.3]),
                quat=np.array([0, 1, 0, 0]),
            )
            self.franka.control_dofs_position(self.qpos[:-2], self.motors_dof)
            self.franka.control_dofs_position(
                np.array([finger_pos, finger_pos]), self.fingers_dof
            )

    def run(self) -> None:
        while self.phase != "end":
            timestamp = self.step_number * self.scene.dt
            self.observe(timestamp)
            self.act(timestamp)
            self.scene.step()
            self.step_number += 1


def main():
    ########################## init ##########################
    gs.init(backend=gs.cpu, precision="32")
    env = Env()
    ############## run the environment #####################
    env.run()


if __name__ == "__main__":
    main()
