"""
Benchmark the reading performance, notably regarding video,
to assess the impact for machine learning application.
Training models requires random access reads to the data.
"""

# 1. Generate a dataset with some video frames, an action type and an observation type.
# 2. Benchmark the reading performance of the dataset with random access.
# 3. Benchmark the reading performance of the dataset with sequential access.

import time
from dataclasses import dataclass
from datetime import timedelta
from fractions import Fraction
from pathlib import Path

import av
import foxglove
import numpy as np
import rerun as rr
from av.video.codeccontext import VideoCodecContext
from foxglove.channels import CompressedVideoChannel, PoseInFrameChannel
from foxglove.schemas import (
    CompressedVideo,
    Pose,
    PoseInFrame,
    Quaternion,
    Timestamp,
    Vector3,
)
from loguru import logger
from mcap.reader import make_reader
from numpy.typing import NDArray
from rerun.components import PoseRotationQuat, PoseTranslation3D
from tqdm import tqdm

EPISODE_DURATION_S = 5
VIDEO_FRAME_RATE_HZ = 30
VIDEO_FRAME_SIZE = (1280, 720)
N_CAMERA_VIEWS = 1

ROBOT_SAMPLING_RATE_HZ = 200  # assume position controlled robot

# Note: this would be cool to make using a simulator like genesis, using a real robot urdf.


def make_video_frame() -> NDArray[np.uint8]:
    # Generate a random image
    frame = np.random.randint(
        0, 255, (VIDEO_FRAME_SIZE[1], VIDEO_FRAME_SIZE[0], 3), dtype=np.uint8
    )
    return frame


def make_codec(
    width: int, height: int, fps: int = VIDEO_FRAME_RATE_HZ
) -> VideoCodecContext:
    codec = av.Codec("libx265", "w")
    codec_ctx = codec.create(kind="video")
    codec_ctx.width = width
    codec_ctx.height = height
    codec_ctx.max_b_frames = 0
    codec_ctx.pix_fmt = "yuv420p"
    codec_ctx.framerate = Fraction(fps, 1)
    codec_ctx.time_base = Fraction(1, fps)
    codec_ctx.options = {"g": "30"}
    return codec_ctx


@dataclass
class Context:
    codec: VideoCodecContext
    timestamps: list[float]


TimestampedPacket = tuple[bytes, float]


@dataclass
class Dataset:
    """In-memory dataset structure."""

    timestamps: NDArray[np.float64]  # Robot timestamps
    positions: NDArray[np.float64]  # Shape (N, 3) for x, y, z
    orientations: NDArray[np.float64]  # Shape (N, 4) for quaternion x, y, z, w
    video_packets: list[TimestampedPacket]  # List of (bytes, float) tuples


def encode_video_frame(
    context: Context, frame: NDArray[np.uint8], timestamp: float
) -> TimestampedPacket | None:
    av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
    context.timestamps.append(timestamp)
    packets = context.codec.encode(av_frame)
    if len(packets) == 1:
        ts = context.timestamps.pop(0)
        msg = (bytes(packets[0]), ts)
        return msg
    return None


def flush_codec(context: Context) -> list[TimestampedPacket]:
    packets = context.codec.encode(None)
    msgs = []
    for packet in packets:
        ts = context.timestamps.pop(0)
        msgs.append((bytes(packet), ts))
    return msgs


def make_dataset() -> Dataset:
    """Generate an in-memory dataset with positions, orientations, and video packets."""
    timestamps = np.linspace(
        0, EPISODE_DURATION_S, int(EPISODE_DURATION_S * ROBOT_SAMPLING_RATE_HZ)
    )
    n_samples = len(timestamps)

    # Initialize arrays for positions and orientations
    # Currently all zeros, but structured as arrays for future use
    positions = np.zeros((n_samples, 3), dtype=np.float64)
    orientations = np.zeros((n_samples, 4), dtype=np.float64)
    orientations[:, 3] = 1.0  # w component of quaternion

    # Encode video frames
    context = Context(
        codec=make_codec(VIDEO_FRAME_SIZE[0], VIDEO_FRAME_SIZE[1]), timestamps=[]
    )
    video_packets: list[TimestampedPacket] = []
    last_frame_timestamp = 0

    for timestamp in tqdm(timestamps, desc="Generating dataset"):
        if (timestamp - last_frame_timestamp) >= (1 / VIDEO_FRAME_RATE_HZ):
            frame = make_video_frame()
            msg = encode_video_frame(context, frame, timestamp)
            if msg is not None:
                video_packets.append(msg)
            last_frame_timestamp = timestamp

    # Flush codec to get remaining packets
    msgs = flush_codec(context)
    video_packets.extend(msgs)

    return Dataset(
        timestamps=timestamps,
        positions=positions,
        orientations=orientations,
        video_packets=video_packets,
    )


def write_mcap_dataset(dataset: Dataset, output_path: Path) -> None:
    """Write dataset to MCAP format."""
    # create a video stream
    video_stream = CompressedVideoChannel(topic="/video")
    # create a pose stream
    action_stream = PoseInFrameChannel(topic="/action")
    # create a observation stream
    observation_stream = PoseInFrameChannel(topic="/observation")

    with foxglove.open_mcap(output_path, allow_overwrite=True):
        # Write robot poses
        for i, timestamp in enumerate(tqdm(dataset.timestamps, desc="Writing MCAP")):
            position = Vector3(
                x=dataset.positions[i, 0],
                y=dataset.positions[i, 1],
                z=dataset.positions[i, 2],
            )
            orientation = Quaternion(
                x=dataset.orientations[i, 0],
                y=dataset.orientations[i, 1],
                z=dataset.orientations[i, 2],
                w=dataset.orientations[i, 3],
            )
            pose = Pose(position=position, orientation=orientation)
            action_stream.log(
                PoseInFrame(pose=pose, timestamp=Timestamp.from_epoch_secs(timestamp)),
                log_time=int(timestamp * 1e9),
            )
            observation_stream.log(
                PoseInFrame(pose=pose, timestamp=Timestamp.from_epoch_secs(timestamp)),
                log_time=int(timestamp * 1e9),
            )

        # Write video packets
        for packet_data, packet_timestamp in dataset.video_packets:
            video_stream.log(
                CompressedVideo(
                    data=packet_data,
                    timestamp=Timestamp.from_epoch_secs(packet_timestamp),
                    format="h265",
                ),
                log_time=int(packet_timestamp * 1e9),
            )


def write_rerun_dataset(dataset: Dataset, output_path: Path) -> None:
    """Write dataset to Rerun format."""
    rr.init("rerun_dataset_benchmark")
    rr.save(output_path)
    # WTF doesn't support av1
    rr.log("video", rr.VideoStream(codec=rr.VideoCodec.H265), static=True)

    # Write robot poses
    for i, timestamp in enumerate(tqdm(dataset.timestamps, desc="Writing Rerun")):
        rr.set_time("robot_time", duration=timestamp)
        rr.log(
            "observation",
            rr.InstancePoses3D(
                translations=PoseTranslation3D(dataset.positions[i]),
                quaternions=PoseRotationQuat(xyzw=dataset.orientations[i]),
            ),
        )
        rr.log(
            "action",
            rr.InstancePoses3D(
                translations=PoseTranslation3D(dataset.positions[i]),
                quaternions=PoseRotationQuat(xyzw=dataset.orientations[i]),
            ),
        )

    # Write video packets
    for packet_data, packet_timestamp in dataset.video_packets:
        rr.set_time("robot_time", duration=packet_timestamp)
        rr.log("video", rr.VideoStream.from_fields(sample=packet_data))


def benchmark_mcap_reader(mcap_path: Path) -> None:
    start_time = time.perf_counter()
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for _ in reader.iter_messages():
            pass
    elapsed = timedelta(seconds=time.perf_counter() - start_time)
    logger.info(f"MCAP reader, sequential read, no decoding: {elapsed}")


def benchmark_rerun_reader(rerun_path: Path) -> None:
    start_time = time.perf_counter()
    recording = rr.dataframe.load_recording(rerun_path)
    view = recording.view(index="robot_time", contents="/**")
    df = view.select().read_pandas()
    for row in df.itertuples():
        pass
    elapsed = timedelta(seconds=time.perf_counter() - start_time)
    logger.info(f"Rerun reader, sequential read, no decoding: {elapsed}")


def main():
    # Generate dataset once
    dataset = make_dataset()

    # Write to both formats
    write_mcap_dataset(dataset, Path("dataset.mcap"))
    benchmark_mcap_reader(Path("dataset.mcap"))

    write_rerun_dataset(dataset, Path("dataset.rrd"))
    benchmark_rerun_reader(Path("dataset.rrd"))


if __name__ == "__main__":
    main()
