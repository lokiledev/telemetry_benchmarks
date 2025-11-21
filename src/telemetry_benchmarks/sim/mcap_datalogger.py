from pathlib import Path

import numpy as np
from foxglove_schemas_protobuf.CameraCalibration_pb2 import CameraCalibration
from foxglove_schemas_protobuf.CompressedVideo_pb2 import CompressedVideo
from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform
from foxglove_schemas_protobuf.FrameTransforms_pb2 import FrameTransforms
from foxglove_schemas_protobuf.Quaternion_pb2 import Quaternion
from foxglove_schemas_protobuf.Vector3_pb2 import Vector3
from genesis.utils.geom import T_to_trans_quat
from google.protobuf.descriptor import FileDescriptor
from google.protobuf.descriptor_pb2 import FileDescriptorSet
from google.protobuf.message import Message as ProtobufMessage
from google.protobuf.timestamp_pb2 import Timestamp
from mcap.writer import Writer

from telemetry_benchmarks.sim.config import CAMERA_RESOLUTION
from telemetry_benchmarks.sim.datalogger import DataLogger, NamedTransform
from telemetry_benchmarks.sim.pb2.joint_state_pb2 import JointState
from telemetry_benchmarks.sim.video_encoder import (
    Context,
    encode_video_frame,
    flush_codec,
    make_codec,
)


def timestamp_to_protobuf(timestamp: float) -> Timestamp:
    """Convert a float timestamp to protobuf Timestamp."""
    seconds = int(timestamp)
    nanos = int((timestamp - seconds) * 1_000_000_000)
    return Timestamp(seconds=seconds, nanos=nanos)


def build_file_descriptor_set(
    message_class: type[ProtobufMessage],
) -> FileDescriptorSet:
    """Recursively build a set of all descriptors necessary to describe a complete message type."""
    file_descriptor_set = FileDescriptorSet()
    seen_dependencies: set[str] = set()

    def append_file_descriptor(file_descriptor: FileDescriptor) -> None:
        for dep in file_descriptor.dependencies:
            if dep.name not in seen_dependencies:
                seen_dependencies.add(dep.name)
                append_file_descriptor(dep)
        file_descriptor.CopyToProto(file_descriptor_set.file.add())

    append_file_descriptor(message_class.DESCRIPTOR.file)
    return file_descriptor_set


class MCAPLogger(DataLogger):
    def __init__(self, output_path: Path):
        self.codec_context = Context(
            codec=make_codec(CAMERA_RESOLUTION[0], CAMERA_RESOLUTION[1]), timestamps=[]
        )
        self.output_path = output_path
        if output_path.exists():
            output_path.unlink()
        self.file = open(output_path, "wb")
        self.mcap_writer = Writer(self.file)
        self.mcap_writer.start()

        video_schema_descriptor = build_file_descriptor_set(CompressedVideo)
        video_schema = self.mcap_writer.register_schema(
            CompressedVideo.DESCRIPTOR.full_name,
            "protobuf",
            video_schema_descriptor.SerializeToString(),
        )
        self.video_stream = self.mcap_writer.register_channel(
            "/video", "protobuf", video_schema
        )

        tf_schema_descriptor = build_file_descriptor_set(FrameTransforms)
        tf_schema = self.mcap_writer.register_schema(
            FrameTransforms.DESCRIPTOR.full_name,
            "protobuf",
            tf_schema_descriptor.SerializeToString(),
        )
        self.frame_transforms_stream = self.mcap_writer.register_channel(
            "/tf", "protobuf", tf_schema
        )
        camera_calibration_schema_descriptor = build_file_descriptor_set(
            CameraCalibration
        )
        camera_calibration_schema_id = self.mcap_writer.register_schema(
            CameraCalibration.DESCRIPTOR.full_name,
            "protobuf",
            camera_calibration_schema_descriptor.SerializeToString(),
        )
        self.camera_calibration_stream = self.mcap_writer.register_channel(
            "/camera_calibration", "protobuf", camera_calibration_schema_id
        )
        joint_state_schema_descriptor = build_file_descriptor_set(JointState)
        self.joint_state_schema = self.mcap_writer.register_schema(
            JointState.DESCRIPTOR.full_name,
            "protobuf",
            joint_state_schema_descriptor.SerializeToString(),
        )
        self.joint_states_stream: dict[str, int] = {}

        # RealSense D455 intrinsic parameters for 640x480 resolution
        # Scaled from typical 1280x720 values: fx=642, fy=641, cx=652.6, cy=360.3
        fx, fy, cx, cy = 321, 427, 326, 240

        calib = CameraCalibration(
            timestamp=timestamp_to_protobuf(0),
            frame_id="camera_link",
            width=CAMERA_RESOLUTION[0],
            height=CAMERA_RESOLUTION[1],
            distortion_model="plumb_bob",
            K=[fx, 0, cx, 0, fy, cy, 0, 0, 1],
            # P is a 3x4 projection matrix (row-major): [fx, 0, cx, Tx, 0, fy, cy, Ty, 0, 0, 1, 0]
            # For monocular cameras: Tx = Ty = 0
            P=[fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0],
        )

        self.mcap_writer.add_message(
            self.camera_calibration_stream,
            log_time=0,
            data=calib.SerializeToString(),
            publish_time=0,
        )

    def add_joint_state_probe(self, joint_name: str) -> None:
        self.joint_states_stream[joint_name] = self.mcap_writer.register_channel(
            f"/joint_states/{joint_name}", "protobuf", self.joint_state_schema
        )

    def log_joint_states(
        self, qpos: np.ndarray, timestamp: float, joint_names: list[str]
    ) -> None:
        if len(self.joint_states_stream) == 0:
            for joint_name in joint_names:
                self.add_joint_state_probe(joint_name)

        for joint, pos in zip(joint_names, qpos):
            self.mcap_writer.add_message(
                self.joint_states_stream[joint],
                log_time=int(timestamp * 1e9),
                data=JointState(
                    position=pos,
                    velocity=0.0,
                    effort=0.0,
                ).SerializeToString(),
                publish_time=int(timestamp * 1e9),
            )

    def log_video(self, video: np.ndarray, timestamp: float) -> None:
        msg = encode_video_frame(self.codec_context, video, timestamp)
        if msg is not None:
            packet_data, packet_timestamp = msg
            payload = CompressedVideo(
                data=packet_data,
                timestamp=timestamp_to_protobuf(packet_timestamp),
                format="av1",
                frame_id="camera_link",
            ).SerializeToString()

            self.mcap_writer.add_message(
                self.video_stream,
                log_time=int(packet_timestamp * 1e9),
                data=payload,
                publish_time=int(packet_timestamp * 1e9),
            )

    def log_end_effector_pose(self, pose: np.ndarray, timestamp: float) -> None:
        pass

    def log_transforms(
        self, transforms: list[NamedTransform], timestamp: float
    ) -> None:
        result = []
        for transform in transforms:
            pos, q = T_to_trans_quat(transform.mat)
            result.append(
                FrameTransform(
                    timestamp=timestamp_to_protobuf(timestamp),
                    parent_frame_id=transform.parent,
                    child_frame_id=transform.child,
                    translation=Vector3(
                        x=pos[0],
                        y=pos[1],
                        z=pos[2],
                    ),
                    rotation=Quaternion(
                        w=q[0],
                        x=q[1],
                        y=q[2],
                        z=q[3],
                    ),
                )
            )
        self.mcap_writer.add_message(
            self.frame_transforms_stream,
            log_time=int(timestamp * 1e9),
            data=FrameTransforms(transforms=result).SerializeToString(),
            publish_time=int(timestamp * 1e9),
        )

    def finish(self) -> None:
        video_msgs = flush_codec(self.codec_context)
        for packet_data, packet_timestamp in video_msgs:
            payload = CompressedVideo(
                data=packet_data,
                timestamp=timestamp_to_protobuf(packet_timestamp),
                format="av1",
                frame_id="camera_link",
            ).SerializeToString()

            self.mcap_writer.add_message(
                self.video_stream,
                log_time=int(packet_timestamp * 1e9),
                data=payload,
                publish_time=int(packet_timestamp * 1e9),
            )
        self.mcap_writer.finish()
        self.file.close()
