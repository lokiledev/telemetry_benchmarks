from dataclasses import dataclass
from fractions import Fraction

import av
import numpy as np
from av.video.codeccontext import VideoCodecContext
from numpy.typing import NDArray


def make_codec(width: int, height: int, fps: int = 30) -> VideoCodecContext:
    codec = av.Codec("libsvtav1", "w")
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
