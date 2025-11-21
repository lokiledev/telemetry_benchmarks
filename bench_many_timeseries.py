"""
Benchmark the performance of recording a single channel a small data type with a high frequency.s
The goal is to see the encoding efficiency and the overall performance of the SDK.
"""

import time
from datetime import timedelta

import numpy as np
import rerun as rr
from loguru import logger
from mcap.writer import Writer
from numpy.typing import NDArray
from tqdm import tqdm

from telemetry_benchmarks.sim.config import OUTPUT_DIR, BenchmarkResult
from telemetry_benchmarks.sim.mcap_datalogger import build_file_descriptor_set
from telemetry_benchmarks.sim.pb2.joint_state_pb2 import JointPositions

SAMPLING_RATE_HZ = 500
DURATION_S = 3600.0

AMPLITUDE = 2.0
FREQUENCY_HZ = 0.1
# Typical humanoid with 20 dof hands
LEG_JOINTS = 6
ARM_JOINTS = 7
TORSO_JOINTS = 2
NECK_JOINTS = 3
HAND_JOINTS = 20

# Total dofs for a complex humanoid robot
N_JOINTS = (
    LEG_JOINTS * 2 + ARM_JOINTS * 2 + TORSO_JOINTS + NECK_JOINTS + HAND_JOINTS * 2
)


def make_data() -> NDArray[np.float32]:
    t = np.linspace(0, DURATION_S, int(DURATION_S * SAMPLING_RATE_HZ))
    result = np.zeros((len(t), N_JOINTS), dtype=np.float32)
    for i, ts in enumerate(t):
        sin = AMPLITUDE * np.sin(2 * np.pi * FREQUENCY_HZ * ts)
        row = np.repeat(sin, N_JOINTS) + np.arange(N_JOINTS)
        result[i] = row.astype(np.float32)
    return np.column_stack((t, result))


def benchmark_foxglove_many_timeseries(data: NDArray[np.float32]) -> BenchmarkResult:
    output_path = OUTPUT_DIR / "many_timeseries.mcap"
    if output_path.exists():
        output_path.unlink()
    file = open(output_path, "wb")
    mcap_writer = Writer(file)
    mcap_writer.start()

    joint_positions_schema = mcap_writer.register_schema(
        JointPositions.DESCRIPTOR.full_name,
        "protobuf",
        build_file_descriptor_set(JointPositions).SerializeToString(),
    )
    joint_positions_channel = mcap_writer.register_channel(
        "/joint_angles", "protobuf", joint_positions_schema
    )
    start_time = time.perf_counter()
    for row in tqdm(data):
        timestamp = row[0]
        msg = JointPositions()
        msg.angles.extend(row[1:])
        mcap_writer.add_message(
            joint_positions_channel,
            log_time=int(timestamp * 1e9),
            data=msg.SerializeToString(),
            publish_time=int(timestamp * 1e9),
        )
    elapsed = timedelta(seconds=time.perf_counter() - start_time)
    mcap_writer.finish()
    file.close()

    mcap_size = (OUTPUT_DIR / "many_timeseries.mcap").stat().st_size
    logger.info(
        f"Foxglove MCAP size: {mcap_size / 1024 / 1024:.2f} MB, write time: {elapsed}"
    )
    return BenchmarkResult(
        output_file=output_path,
        size_mb=mcap_size / 1024 / 1024,
        duration=elapsed,
    )


def benchmark_rerun_many_timeseries(data: NDArray[np.float32]) -> BenchmarkResult:
    rr.init("many_timeseries")
    rr.save(OUTPUT_DIR / "many_timeseries.rrd")
    start_time = time.perf_counter()
    for row in tqdm(data):
        timestamp = row[0]
        rr.set_time("joint_angles", duration=timestamp)
        rr.log("joint_angles", rr.Scalars(row[1:]))
    elapsed = timedelta(seconds=time.perf_counter() - start_time)
    rr.init("many_timeseries")  # force flush
    rrd_size = (OUTPUT_DIR / "many_timeseries.rrd").stat().st_size
    logger.info(f"Rerun size: {rrd_size / 1024 / 1024:.2f} MB, write time: {elapsed}")
    return BenchmarkResult(
        output_file=OUTPUT_DIR / "many_timeseries.rrd",
        size_mb=rrd_size / 1024 / 1024,
        duration=elapsed,
    )


def benchmark_rerun_many_timeseries_column(
    data: NDArray[np.float32],
) -> BenchmarkResult:
    rr.init("many_timeseries_column")
    rr.save(OUTPUT_DIR / "many_timeseries_column.rrd")
    start_time = time.perf_counter()
    columns = rr.Scalars.columns(scalars=data[:, 1:])
    rr.send_columns(
        "joint_angles",
        indexes=[rr.TimeColumn("joint_angles", duration=data[:, 0])],
        columns=columns,
    )
    elapsed = timedelta(seconds=time.perf_counter() - start_time)
    rr.init("many_timeseries_column")  # force flush
    rrd_size = (OUTPUT_DIR / "many_timeseries_column.rrd").stat().st_size
    logger.info(f"Rerun size: {rrd_size / 1024 / 1024:.2f} MB, write time: {elapsed}")
    return BenchmarkResult(
        output_file=OUTPUT_DIR / "many_timeseries_column.rrd",
        size_mb=rrd_size / 1024 / 1024,
        duration=elapsed,
    )


def run_benchmark_many_timeseries() -> tuple[BenchmarkResult, BenchmarkResult]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = make_data()
    foxglove_result = benchmark_foxglove_many_timeseries(data)
    rerun_result = benchmark_rerun_many_timeseries(data)
    rerun_column_result = benchmark_rerun_many_timeseries_column(data)
    return foxglove_result, rerun_result, rerun_column_result


if __name__ == "__main__":
    run_benchmark_many_timeseries()
