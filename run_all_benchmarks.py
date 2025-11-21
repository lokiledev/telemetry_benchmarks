from bench_many_timeseries import run_benchmark_many_timeseries
from bench_reader import run_benchmark_reader
from bench_single_pointcloud import run_benchmark_single_pointcloud
from bench_single_timeseries import run_benchmark_single_timeseries
from telemetry_benchmarks.sim.config import OUTPUT_DIR


def print_results_table(results):
    """Print benchmark results in a markdown table format."""
    print("\n## Benchmark Results\n")
    print("| Benchmark | Format | File Size (MB) | Duration |")
    print("|-----------|--------|----------------|----------|")

    # Many timeseries
    foxglove_many, rerun_many, rerun_column_many = results[0]
    print(
        f"| Many Timeseries | Foxglove MCAP | {foxglove_many.size_mb:.2f} | {foxglove_many.duration} |"
    )
    print(
        f"| Many Timeseries | Rerun RRD | {rerun_many.size_mb:.2f} | {rerun_many.duration} |"
    )
    print(
        f"| Many Timeseries | Rerun RRD (Column) | {rerun_column_many.size_mb:.2f} | {rerun_column_many.duration} |"
    )
    # Single timeseries
    foxglove_single, rerun_single, rerun_column_single = results[1]
    print(
        f"| Single Timeseries | Foxglove MCAP | {foxglove_single.size_mb:.2f} | {foxglove_single.duration} |"
    )
    print(
        f"| Single Timeseries | Rerun RRD | {rerun_single.size_mb:.2f} | {rerun_single.duration} |"
    )
    print(
        f"| Single Timeseries | Rerun RRD (Column) | {rerun_column_single.size_mb:.2f} | {rerun_column_single.duration} |"
    )

    # Single pointcloud
    foxglove_pc, rerun_pc, rerun_column_pc = results[2]
    print(
        f"| Single Pointcloud | Foxglove MCAP | {foxglove_pc.size_mb:.2f} | {foxglove_pc.duration} |"
    )
    print(
        f"| Single Pointcloud | Rerun RRD | {rerun_pc.size_mb:.2f} | {rerun_pc.duration} |"
    )
    print(
        f"| Single Pointcloud | Rerun RRD (Column) | {rerun_column_pc.size_mb:.2f} | {rerun_column_pc.duration} |"
    )

    # Reader
    mcap_reader, mcap_random_access_reader, rerun_reader, rerun_random_access_reader = (
        results[3]
    )
    print(
        f"| Reader | Foxglove MCAP | {mcap_reader.size_mb:.2f} | {mcap_reader.duration} |"
    )
    print(
        f"| Reader | Foxglove MCAP (Random Access) | {mcap_random_access_reader.size_mb:.2f} | {mcap_random_access_reader.duration} |"
    )
    print(
        f"| Reader | Rerun RRD | {rerun_reader.size_mb:.2f} | {rerun_reader.duration} |"
    )
    print(
        f"| Reader | Rerun RRD (Random Access) | {rerun_random_access_reader.size_mb:.2f} | {rerun_random_access_reader.duration} |"
    )
    print()


def run_all_benchmarks() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    many_timeseries_result = run_benchmark_many_timeseries()
    single_timeseries_result = run_benchmark_single_timeseries()
    single_pointcloud_result = run_benchmark_single_pointcloud()
    reader_result = run_benchmark_reader()
    results = (
        many_timeseries_result,
        single_timeseries_result,
        single_pointcloud_result,
        reader_result,
    )
    print_results_table(results)


if __name__ == "__main__":
    run_all_benchmarks()
