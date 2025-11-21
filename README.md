# Telemetry Benchmarks

## Overview

This project provides a systematic comparison between [Foxglove](https://foxglove.dev/) (MCAP format) and [Rerun](https://rerun.io/) (RRD format) for robotics telemetry data recording and playback. The benchmarks cover:

- **SDK Performance**: Write and read performance of both SDKs
- **File Storage**: Compression efficiency and file size comparisons
- **Viewer Performance**: Playback and visualization capabilities

## Features

### Benchmark Suite

The project includes several benchmark scripts:

- **`bench_reader.py`**: Generates a representative fake dataset and benchmarks reading performance with video frames, poses, and random/sequential access patterns
- **`bench_single_pointcloud.py`**: Evaluates point cloud recording and playback performance
- **`bench_single_timeseries.py`**: Tests high-frequency time series data recording efficiency

### Robot Simulation

The project includes a self-contained simulation package (`src/telemetry_benchmarks/sim/`) based on [genesis-world](https://genesis-world.readthedocs.io/en/latest/#) featuring:

- **Robot Arm Simulation**: A realistic robot arm simulation using the SO101 robot model
- **Dual Format Recording**: Simultaneously records data in both MCAP (Foxglove) and RRD (Rerun) formats
- **Realistic Data Generation**: Produces meaningful telemetry data including:
  - Joint states
  - End-effector poses
  - Transform trees
  - Video stream from simulated camera

This simulation serves as both a demo and a source of realistic benchmark data, ensuring that performance comparisons are based on representative robotics workloads.

## Installation

### Prerequisites

- Python >= 3.10
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

1. **Create a virtual environment and install dependencies:**

   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv sync
   ```

2. **Special Installation for Rerun (AV1 Codec Support)**

   The latest Rerun release doesn't include AV1 codec support yet. To enable AV1 encoding/decoding, install a pre-release build:

   ```bash
   uv pip install --pre --no-index -f https://build.rerun.io/commit/0c88e94/wheels --upgrade rerun-sdk
   ```

   > **Note**: This installs a pre-release build from a specific commit. Once AV1 support is available in the official release, this step can be skipped.

## Usage

### Running Benchmarks

Run individual benchmark scripts:

```bash
# Benchmark reader performance with video and pose data
python bench_reader.py

# Benchmark point cloud recording
python bench_single_pointcloud.py

# Benchmark high-frequency time series data
python bench_single_timeseries.py
```

### Running the Robot Simulation

The simulation can be run with different logging backends:

```bash
# Run simulation with MCAP logging (Foxglove)
python -m telemetry_benchmarks.sim.robot_arm_sim --logger mcap

# Run simulation with Rerun logging
python -m telemetry_benchmarks.sim.robot_arm_sim --logger rerun
```

### Output Files

All generated files are saved to the `output/` directory:

- `*.mcap` - Foxglove MCAP format files
- `*.rrd` - Rerun format files

## Project Structure

```
telemetry_benchmarks/
├── bench_reader.py              # Main reader benchmark
├── bench_single_pointcloud.py    # Point cloud benchmark
├── bench_single_timeseries.py    # Time series benchmark
├── src/
│   └── telemetry_benchmarks/
│       └── sim/                  # Simulation package
│           ├── config.py          # Configuration
│           ├── datalogger.py     # Abstract logger interface
│           ├── mcap_datalogger.py # Foxglove MCAP logger
│           ├── rerun_datalogger.py # Rerun logger
│           ├── robot_arm_sim.py  # Robot simulation
│           └── SO101/            # Robot model assets
└── output/                       # Generated benchmark files
```

## Viewer tests

* To open the files with foxglove:
```sh
foxglove-studio output/robot_arm.mcap
```

* To open the files with rerun:
```sh
rerun output/robot_arm.rrd
```

## Results

| Benchmark | Format | File Size (MB) | Duration |
|-----------|--------|----------------|----------|
| Many Timeseries | Foxglove MCAP | 519.37 | 0:00:29.356378 |
| Many Timeseries | Rerun RRD | 679.92 | 0:00:56.065598 |
| Many Timeseries | Rerun RRD (Column) | 652.74 | 0:00:00.224108 |
| Single Timeseries | Foxglove MCAP | 169.81 | 0:00:10.606665 |
| Single Timeseries | Rerun RRD | 89.81 | 0:01:51.583999 |
| Single Timeseries | Rerun RRD (Column) | 131.28 | 0:00:00.136080 |
| Single Pointcloud | Foxglove MCAP | 1563.69 | 0:00:07.199298 |
| Single Pointcloud | Rerun RRD | 829.41 | 0:00:02.564451 |
| Single Pointcloud | Rerun RRD (Column) | 828.45 | 0:00:00.191245 |
| Reader | Foxglove MCAP | 669.44 | 0:00:00.845047 |
| Reader | Foxglove MCAP (Random Access) | 669.44 | 0:00:02.246469 |
| Reader | Rerun RRD | 673.04 | 0:00:01.513706 |
| Reader | Rerun RRD (Random Access) | 673.04 | 0:00:01.070348 |

## Contributing

This project is designed to provide objective performance comparisons. When adding new benchmarks, ensure:

- Both formats are tested under identical conditions
- Data generation is deterministic and reproducible
- Results are clearly documented

# Updating protobuf messages

If you need to create new protobuf messages for foxglove,
you will need the buf tool to generate them.

* install [buf](https://buf.build/docs/cli/installation/)
* run `buf generate`
This will generate new files in `src/telemetry_benchmarks/proto/pb2`

* commit the generated files.

