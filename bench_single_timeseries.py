"""
Benchmark the performance of recording a single channel a small data type with a high frequency.s
The goal is to see the encoding efficiency and the overall performance of the SDK.
"""

import os
import time
from datetime import timedelta

import foxglove
import numpy as np
import rerun as rr
from foxglove.channels import Point3Channel, PointCloudChannel
from foxglove.schemas import Point3, PointCloud
from loguru import logger
from numpy.typing import NDArray
from rerun.components import Position3D, Position3DBatch

SAMPLING_RATE_HZ = 1000
DURATION_S = 3600

AMPLITUDE = 2.0
FREQUENCY_HZ = 0.1


def make_data() -> NDArray[np.float64]:
    t = np.linspace(0, DURATION_S, int(DURATION_S * SAMPLING_RATE_HZ))
    x = AMPLITUDE * np.sin(2 * np.pi * 1 * FREQUENCY_HZ * t)
    y = AMPLITUDE * np.sin(2 * np.pi * 2 * FREQUENCY_HZ * t)
    z = AMPLITUDE * np.sin(2 * np.pi * 3 * FREQUENCY_HZ * t)
    return np.column_stack((t, x, y, z))


def make_pointcloud() -> NDArray[np.float64]:
    sampling_rate_hz = 20
    duration_s = 3600
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


def benchmark_foxglove_single_timeseries(data: NDArray[np.float64]) -> None:
    channel = Point3Channel(topic="/log1")
    with foxglove.open_mcap("single_timeseries.mcap", allow_overwrite=True):
        start_time = time.perf_counter()
        for timestamp, x, y, z in data:
            channel.log(
                Point3(x=x, y=y, z=z),
                log_time=int(timestamp * 1e9),
            )
        elapsed = timedelta(seconds=time.perf_counter() - start_time)
    mcap_size = os.path.getsize("single_timeseries.mcap")
    logger.info(
        f"Foxglove MCAP size: {mcap_size / 1024 / 1024:.2f} MB, write time: {elapsed}"
    )


def benchmark_rerun_single_timeseries(data: NDArray[np.float64]) -> None:
    rr.init("single_timeseries")
    rr.save("single_timeseries.rrd")
    start_time = time.perf_counter()
    for timestamp, x, y, z in data:
        rr.set_time("log_time", timestamp=timestamp)
        rr.log("log1", rr.AnyValues(position=Position3D([x, y, z])))
    elapsed = timedelta(seconds=time.perf_counter() - start_time)
    rrd_size = os.path.getsize("single_timeseries.rrd")
    logger.info(f"Rerun size: {rrd_size / 1024 / 1024:.2f} MB, write time: {elapsed}")


def benchmark_rerun_column(data: NDArray[np.float64]) -> None:
    rr.init("single_timeseries_column")
    rr.save("single_timeseries_column.rrd")
    positions = Position3DBatch(data[:, 1:])
    start_time = time.perf_counter()
    # AnyValues.columns() returns a ComponentColumnList which is iterable
    # We need to unpack it to get individual ComponentColumn objects
    columns = rr.AnyValues.columns(position=positions)
    rr.send_columns(
        "log1",
        indexes=[rr.TimeColumn("log_time", timestamp=data[:, 0])],
        columns=columns,  # Pass the ComponentColumnList directly (it's iterable)
    )
    elapsed = timedelta(seconds=time.perf_counter() - start_time)
    rr.init("single_timeseries_column")  # force flush
    rrd_size = os.path.getsize("single_timeseries_column.rrd")
    logger.info(f"Rerun size: {rrd_size / 1024 / 1024:.2f} MB, write time: {elapsed}")


def benchmark_foxglove_pointcloud(data: NDArray[np.float64]) -> None:
    channel = PointCloudChannel(topic="/log1")
    with foxglove.open_mcap("pointcloud.mcap", allow_overwrite=True):
        start_time = time.perf_counter()
        for timestamp, x, y, z in data:
            channel.log(PointCloud(positions=Position3D([x, y, z])))
        elapsed = timedelta(seconds=time.perf_counter() - start_time)
    mcap_size = os.path.getsize("pointcloud.mcap")
    logger.info(
        f"Foxglove MCAP size: {mcap_size / 1024 / 1024:.2f} MB, write time: {elapsed}"
    )


def benchmark_rerun_pointcloud(data: NDArray[np.float64]) -> None:
    rr.init("pointcloud")
    rr.save("pointcloud.rrd")
    start_time = time.perf_counter()
    for pointcloud_row in data:
        rr.set_time("log_time", timestamp=pointcloud_row[0])
        rr.log("log1", rr.Points3D(pointcloud_row[1:]))
    elapsed = timedelta(seconds=time.perf_counter() - start_time)
    rrd_size = os.path.getsize("pointcloud.rrd")
    logger.info(f"Rerun size: {rrd_size / 1024 / 1024:.2f} MB, write time: {elapsed}")


def main():
    data = make_data()
    # benchmark_foxglove_single_timeseries(data)
    # benchmark_rerun_single_timeseries(data)
    # benchmark_rerun_column(data)
    points = make_pointcloud()
    # benchmark_foxglove_pointcloud(points)
    benchmark_rerun_pointcloud(points)


if __name__ == "__main__":
    main()
