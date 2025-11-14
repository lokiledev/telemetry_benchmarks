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
from foxglove.channels import Point3Channel
from foxglove.schemas import Point3
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
        rr.set_time("log1", duration=timestamp)
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
        indexes=[rr.TimeColumn("log1", duration=data[:, 0])],
        columns=columns,  # Pass the ComponentColumnList directly (it's iterable)
    )
    elapsed = timedelta(seconds=time.perf_counter() - start_time)
    rr.init("single_timeseries_column")  # force flush
    rrd_size = os.path.getsize("single_timeseries_column.rrd")
    logger.info(f"Rerun size: {rrd_size / 1024 / 1024:.2f} MB, write time: {elapsed}")


def main():
    data = make_data()
    benchmark_foxglove_single_timeseries(data)
    benchmark_rerun_single_timeseries(data)
    benchmark_rerun_column(data)


if __name__ == "__main__":
    main()
