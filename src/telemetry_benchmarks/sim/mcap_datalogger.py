from pathlib import Path

import foxglove
import numpy as np
from foxglove.channels import CompressedVideoChannel, FrameTransformsChannel
from foxglove.schemas import (
    CompressedVideo,
    FrameTransform,
    FrameTransforms,
    Quaternion,
    Timestamp,
    Vector3,
)
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from telemetry_benchmarks.sim.config import CAMERA_RESOLUTION
from telemetry_benchmarks.sim.datalogger import DataLogger, NamedTransform
from telemetry_benchmarks.sim.video_encoder import (
    Context,
    encode_video_frame,
    flush_codec,
    make_codec,
)


def rot_mat_to_quaternion(rot_mat: NDArray[np.float64]) -> Quaternion:
    q = Rotation.from_matrix(rot_mat[:3, :3]).as_quat()
    return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])


class MCAPLogger(DataLogger):
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.mcap_writer = foxglove.open_mcap(output_path, allow_overwrite=True)
        self.codec_context = Context(
            codec=make_codec(CAMERA_RESOLUTION[0], CAMERA_RESOLUTION[1]), timestamps=[]
        )
        self.video_stream = CompressedVideoChannel(topic="/video")
        self.frame_transforms_stream = FrameTransformsChannel(topic="/tf")

    def log_joint_states(self, qpos: np.ndarray, timestamp: float) -> None:
        pass

    def log_video(self, video: np.ndarray, timestamp: float) -> None:
        msg = encode_video_frame(self.codec_context, video, timestamp)
        if msg is not None:
            packet_data, packet_timestamp = msg
            self.video_stream.log(
                CompressedVideo(
                    data=packet_data,
                    timestamp=Timestamp.from_epoch_secs(packet_timestamp),
                    format="av1",
                ),
                log_time=int(packet_timestamp * 1e9),
            )

    def log_end_effector_pose(self, pose: np.ndarray, timestamp: float) -> None:
        pass

    def log_transforms(
        self, transforms: list[NamedTransform], timestamp: float
    ) -> None:
        result = []
        for transform in transforms:
            breakpoint
            result.append(
                FrameTransform(
                    timestamp=Timestamp.from_epoch_secs(timestamp),
                    parent_frame_id=transform.parent,
                    child_frame_id=transform.child,
                    translation=Vector3(
                        x=transform.mat[0, 3],
                        y=transform.mat[1, 3],
                        z=transform.mat[2, 3],
                    ),
                    rotation=rot_mat_to_quaternion(transform.mat),
                )
            )
        self.frame_transforms_stream.log(
            FrameTransforms(transforms=result),
            log_time=int(timestamp * 1e9),
        )

    def finish(self) -> None:
        video_msgs = flush_codec(self.codec_context)
        for packet_data, packet_timestamp in video_msgs:
            self.video_stream.log(
                CompressedVideo(
                    data=packet_data,
                    timestamp=Timestamp.from_epoch_secs(packet_timestamp),
                    format="av1",
                ),
                log_time=int(packet_timestamp * 1e9),
            )
        self.mcap_writer.close()
