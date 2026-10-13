"""Run the real entrypoint in the built image, with a fake `vllm` that prints its arguments."""

import subprocess

import pytest

pytestmark = pytest.mark.functional

FAKE_VLLM = 'mkdir -p /tmp/fake && printf \'#!/bin/sh\\necho "$@"\\n\' > /tmp/fake/vllm && chmod +x /tmp/fake/vllm'


def entrypoint(image: str, env: dict[str, str], *args: str) -> subprocess.CompletedProcess:
    flags = [f"--env={k}={v}" for k, v in env.items()]
    script = f'{FAKE_VLLM} && PATH=/tmp/fake:$PATH exec /usr/local/bin/entrypoint.sh "$@"'
    return subprocess.run(
        ["docker", "run", "--rm", *flags, "--entrypoint", "bash", image, "-c", script, "entrypoint", *args],
        capture_output=True, text=True, timeout=120,
    )


def test_requires_model(docker_image):
    result = entrypoint(docker_image, {})
    assert result.returncode == 1
    assert "MODEL is not set" in result.stderr


def test_s3_model_uses_streamer(docker_image):
    result = entrypoint(docker_image, {"MODEL": "s3://models/org/m"})
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == [
        "serve", "s3://models/org/m", "--host", "0.0.0.0", "--port", "8000",
        "--gpu-memory-utilization", "0.90", "--load-format", "runai_streamer",
    ]


def test_hugging_face_model_has_no_streamer(docker_image):
    result = entrypoint(docker_image, {"MODEL": "org/m"})
    assert "--load-format" not in result.stdout.split()


def test_settings_map_to_flags(docker_image):
    env = {
        "MODEL": "org/m",
        "MAX_MODEL_LEN": "4096",
        "SERVED_MODEL_NAME": "name",
        "VLLM_API_KEY": "sk-1",
        "GPU_MEMORY_UTILIZATION": "0.5",
        "EXTRA_ARGS": "--enable-prefix-caching --seed 1",
    }
    args = entrypoint(docker_image, env, "--trailing").stdout.split()
    for flag, value in [("--max-model-len", "4096"), ("--served-model-name", "name"),
                        ("--api-key", "sk-1"), ("--gpu-memory-utilization", "0.5"), ("--seed", "1")]:
        assert args[args.index(flag) + 1] == value
    assert "--enable-prefix-caching" in args
    assert args[-1] == "--trailing"


def test_model_store_runs_in_image(docker_image):
    result = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "python3", docker_image, "/opt/model_store.py", "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "seed" in result.stdout and "list" in result.stdout and "rm" in result.stdout
