from pathlib import Path

import numpy as np
import rerun as rr
from genesis.utils.geom import T_to_trans_quat

from telemetry_benchmarks.sim.config import CAMERA_RESOLUTION
from telemetry_benchmarks.sim.datalogger import DataLogger, NamedTransform
from telemetry_benchmarks.sim.video_encoder import (
    Context,
    encode_video_frame,
    flush_codec,
    make_codec,
)


class RerunLogger(DataLogger):
    def __init__(self, output_path: Path, urdf_path: Path):
        rr.init("robot_arm_demo")
        rr.save(output_path)
        rr.log("video", rr.VideoStream(codec=rr.VideoCodec.AV1), static=True)
        rr.log_file_from_path(urdf_path, static=True)
        self.output_path = output_path
        self.codec_context = Context(
            codec=make_codec(CAMERA_RESOLUTION[0], CAMERA_RESOLUTION[1]), timestamps=[]
        )
        # RealSense D455 intrinsic parameters for 640x480 resolution
        # Scaled from typical 1280x720 values: fx=642, fy=641, cx=652.6, cy=360.3
        # fx, fy, cx, cy = 321, 427, 326, 240

        # calib = CameraCalibration(
        #     timestamp=Timestamp.from_epoch_secs(0),
        #     frame_id="camera_link",
        #     width=CAMERA_RESOLUTION[0],
        #     height=CAMERA_RESOLUTION[1],
        #     distortion_model="plumb_bob",
        #     D=[0, 0, 0, 0, 0],
        #     K=[fx, 0, cx, 0, fy, cy, 0, 0, 1],
        #     # P is a 3x4 projection matrix (row-major): [fx, 0, cx, Tx, 0, fy, cy, Ty, 0, 0, 1, 0]
        #     # For monocular cameras: Tx = Ty = 0
        #     P=[fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0],
        # )

    def log_joint_states(self, qpos: np.ndarray, timestamp: float) -> None:
        pass

    def log_video(self, video: np.ndarray, timestamp: float) -> None:
        msg = encode_video_frame(self.codec_context, video, timestamp)
        if msg is not None:
            packet_data, packet_timestamp = msg
            rr.set_time("sim_time", duration=packet_timestamp)
            rr.log("video", rr.VideoStream.from_fields(sample=packet_data))

    def log_end_effector_pose(self, pose: np.ndarray, timestamp: float) -> None:
        pass

    def log_transforms(
        self, transforms: list[NamedTransform], timestamp: float
    ) -> None:
        rr.set_time("sim_time", duration=timestamp)
        for named_tf in transforms:
            pos, quat = T_to_trans_quat(named_tf.mat)
            rr.log(
                "transforms",
                rr.Transform3D(
                    translation=pos,
                    quaternion=quat,
                    parent_frame=named_tf.parent,
                    child_frame=named_tf.child,
                    relation=rr.TransformRelation.ChildFromParent,
                ),
            )

    def finish(self) -> None:
        video_msgs = flush_codec(self.codec_context)
        for packet_data, packet_timestamp in video_msgs:
            rr.set_time("sim_time", duration=packet_timestamp)
            rr.log("video", rr.VideoStream.from_fields(sample=packet_data))
