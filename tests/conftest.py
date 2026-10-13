import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
IMAGE = "host-your-own-llm/vllm:latest"


def kill_tree(process: subprocess.Popen) -> None:
    """Kill a process and its children (`uv run` starts the CLI as a child process)."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True)
    else:
        os.killpg(process.pid, signal.SIGKILL)


def docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0


def nvidia_runtime() -> bool:
    info = subprocess.run(["docker", "info", "--format", "{{json .Runtimes}}"], capture_output=True, text=True)
    return "nvidia" in info.stdout


@pytest.fixture(scope="session")
def docker_image():
    if not docker_available():
        pytest.skip(f"needs Docker and the built image ({IMAGE}); run `docker compose build`")
    return IMAGE


class Project:
    """An isolated copy of the stack: its own compose project, ports, volume and .env."""

    def __init__(self, path: Path, name: str, api_port: int, s3_port: int):
        self.path = path
        self.name = name
        self.api_port = api_port
        self.s3_port = s3_port
        self.volume = f"{name}_model-store"
        shutil.copy(ROOT / "docker-compose.yml", path)
        shutil.copytree(ROOT / "image", path / "image")

    @property
    def env_path(self) -> Path:
        return self.path / ".env"

    def write_env(self, **values: str) -> None:
        base = {"COMPOSE_PROJECT_NAME": self.name, "API_PORT": str(self.api_port),
                "S3_PORT": str(self.s3_port), "S3_CONSOLE_PORT": str(self.s3_port + 1)}
        self.env_path.write_text("".join(f"{k}={v}\n" for k, v in {**base, **values}.items()))

    def env(self) -> dict[str, str]:
        from host_your_own_llm.adapters.envfile import read_env
        return read_env(self.env_path)

    def llm(self, *args: str, input: str | None = None, timeout: int = 600) -> subprocess.CompletedProcess:
        """Run the real CLI inside this project. A command still running after `timeout` fails the test."""
        cmd = ["uv", "run", "--project", str(ROOT), "llm", *args]
        process = subprocess.Popen(
            cmd, cwd=self.path, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
            env={**os.environ, "COLUMNS": "200", "NO_COLOR": "1", "PYTHONIOENCODING": "utf-8"},
            start_new_session=sys.platform != "win32",
        )
        try:
            stdout, stderr = process.communicate(input or "", timeout=timeout)
        except subprocess.TimeoutExpired:
            kill_tree(process)
            stdout, stderr = process.communicate()
            pytest.fail(f"`llm {' '.join(args)}` was still running after {timeout}s\n{stdout}{stderr}")
        return subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)

    def compose(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["docker", "compose", "-p", self.name, *args], cwd=self.path, capture_output=True, text=True)

    def containers(self) -> list[str]:
        return self.compose("ps", "-q").stdout.split()

    def volume_exists(self) -> bool:
        return self.volume in subprocess.run(["docker", "volume", "ls", "-q"], capture_output=True, text=True).stdout.split()

    def s3_client(self):
        """A client on the host, for checking what is really in the store."""
        import boto3
        env = self.env()
        return boto3.client("s3", endpoint_url=f"http://localhost:{self.s3_port}", region_name="us-east-1",
                            aws_access_key_id=env["S3_ACCESS_KEY"], aws_secret_access_key=env["S3_SECRET_KEY"])

    def bucket_keys(self, prefix: str = "") -> list[str]:
        s3 = self.s3_client()
        try:
            pages = s3.get_paginator("list_objects_v2").paginate(Bucket=self.env().get("S3_BUCKET", "models"), Prefix=prefix)
            return sorted(obj["Key"] for page in pages for obj in page.get("Contents", []))
        except s3.exceptions.NoSuchBucket:
            return []

    def destroy(self) -> None:
        self.compose("down", "-v", "--remove-orphans")


def make_project(tmp_path_factory, name: str, api_port: int, s3_port: int):
    if not docker_available():
        pytest.skip(f"needs Docker and the built image ({IMAGE}); run `docker compose build`")
    project = Project(tmp_path_factory.mktemp(name), name, api_port, s3_port)
    project.destroy()  # clear leftovers from an aborted earlier run
    # Build from the copied source so the tests exercise the current code (cached, so usually seconds)
    build = project.compose("build", "-q", "vllm")
    assert build.returncode == 0, build.stdout + build.stderr
    return project
