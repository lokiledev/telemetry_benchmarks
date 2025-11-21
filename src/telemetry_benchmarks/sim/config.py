from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

OUTPUT_DIR = Path("output")
CAMERA_RESOLUTION = (640, 480)
CAMERA_FPS = 60


@dataclass
class BenchmarkResult:
    output_file: Path
    size_mb: float
    duration: timedelta
