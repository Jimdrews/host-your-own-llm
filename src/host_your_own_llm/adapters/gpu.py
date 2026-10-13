import subprocess

from ..domain import GpuInfo
from .compose import Runner


def detect_gpu(runner: Runner = subprocess.run) -> GpuInfo | None:
    try:
        out = runner(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout.splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return None
    name, mib = out.rsplit(",", 1)
    return GpuInfo(name=name.strip(), vram_gb=int(mib) / 1024)
