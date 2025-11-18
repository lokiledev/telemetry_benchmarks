"""
Benchmark the performance of recording a representative point cloud channel.
"""

import time
from datetime import timedelta

import foxglove
import numpy as np
import rerun as rr
from foxglove.channels import PointCloudChannel
from foxglove.schemas import (
    PackedElementField,
    PackedElementFieldNumericType,
    PointCloud,
)
from loguru import logger
from numpy.typing import NDArray

from config import OUTPUT_DIR

SAMPLING_RATE_HZ = 1000
DURATION_S = 3600

AMPLITUDE = 2.0
FREQUENCY_HZ = 0.1


def make_pointcloud() -> NDArray[np.float64]:
    sampling_rate_hz = 20.0
    duration_s = 3600.0
    n_points = 1000

    t = np.linspace(0, duration_s, int(duration_s * sampling_rate_hz))
    data = []

    # For distributing points uniformly on the sphere: use spherical Fibonacci point set
    phi = np.arccos(1 - 2 * (np.arange(n_points) + 0.5) / n_points)
    theta = np.pi * (1 + 5**0.5) * (np.arange(n_points) + 0.5)

    for timestamp in t:
        # Evolving radius follows a sinusoidal pattern over time
        radius = AMPLITUDE * (1.0 + 0.5 * np.sin(2 * np.pi * FREQUENCY_HZ * timestamp))
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        # Each row: [timestamp, x0, y0, z0, x1, y1, z1, ..., xN, yN, zN]
        pointcloud_row = [timestamp]
        # Stack the coordinates for this timestep
        for xi, yi, zi in zip(x, y, z):
            pointcloud_row.extend([xi, yi, zi])
        data.append(np.array(pointcloud_row))
    return np.array(data)


def benchmark_foxglove_pointcloud(data: NDArray[np.float64]) -> None:
    channel = PointCloudChannel(topic="/log1")
    with foxglove.open_mcap(OUTPUT_DIR / "pointcloud.mcap", allow_overwrite=True):
        start_time = time.perf_counter()
        fields = [
            PackedElementField(
                name="x", offset=0, type=PackedElementFieldNumericType.Float64
            ),
            PackedElementField(
                name="y", offset=8, type=PackedElementFieldNumericType.Float64
            ),
            PackedElementField(
                name="z", offset=16, type=PackedElementFieldNumericType.Float64
            ),
        ]
        for pointcloud_row in data:
            channel.log(
                PointCloud(
                    fields=fields,
                    point_stride=8 * 3,
                    data=bytes(pointcloud_row[1:].data),
                ),
                log_time=int(pointcloud_row[0] * 1e9),
            )
        elapsed = timedelta(seconds=time.perf_counter() - start_time)
    mcap_size = (OUTPUT_DIR / "pointcloud.mcap").stat().st_size
    logger.info(
        f"Foxglove MCAP size: {mcap_size / 1024 / 1024:.2f} MB, write time: {elapsed}"
    )


def benchmark_rerun_pointcloud(data: NDArray[np.float64]) -> None:
    rr.init("pointcloud")
    rr.save(OUTPUT_DIR / "pointcloud.rrd")
    start_time = time.perf_counter()
    for pointcloud_row in data:
        rr.set_time("log1", duration=pointcloud_row[0])
        rr.log("log1", rr.Points3D(pointcloud_row[1:]))
    elapsed = timedelta(seconds=time.perf_counter() - start_time)
    rrd_size = (OUTPUT_DIR / "pointcloud.rrd").stat().st_size
    logger.info(f"Rerun size: {rrd_size / 1024 / 1024:.2f} MB, write time: {elapsed}")


def benchmark_rerun_pointcloud_column(data: NDArray[np.float64]) -> None:
    rr.init("pointcloud_column")
    rr.save(OUTPUT_DIR / "pointcloud_column.rrd")

    timestamps = data[:, 0]
    n_points = (data.shape[1] - 1) // 3  # Subtract 1 for timestamp column
    # We want shape (N, n_points, 3)
    points = data[:, 1:].reshape(-1, n_points, 3)

    start_time = time.perf_counter()
    columns = rr.Points3D.columns(positions=points)
    rr.send_columns(
        "log1",
        indexes=[rr.TimeColumn("log1", duration=timestamps)],
        columns=columns,
    )
    elapsed = timedelta(seconds=time.perf_counter() - start_time)
    rr.init("pointcloud_column")  # force flush
    rrd_size = (OUTPUT_DIR / "pointcloud_column.rrd").stat().st_size
    logger.info(f"Rerun size: {rrd_size / 1024 / 1024:.2f} MB, write time: {elapsed}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    points = make_pointcloud()
    benchmark_foxglove_pointcloud(points)
    benchmark_rerun_pointcloud(points)
    benchmark_rerun_pointcloud_column(points)


if __name__ == "__main__":
    main()
