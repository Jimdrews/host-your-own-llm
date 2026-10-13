"""Serving on the GPU, in an isolated project. Opt in with `uv run pytest -m gpu`.

Needs the GPU free (stop your own stack first) and downloads about 1.3 GB into the test volume.
The tests run in order and share one project.
"""

import json
import subprocess
import time
import urllib.error
import urllib.request

import pytest

from conftest import IMAGE, make_project, nvidia_runtime

pytestmark = pytest.mark.gpu

FIRST = "Qwen/Qwen2.5-0.5B-Instruct"
SECOND = "HuggingFaceTB/SmolLM2-135M-Instruct"
STANDALONE_PORT = 18001


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    if not nvidia_runtime():
        pytest.skip("Docker doesn't list the NVIDIA runtime")
    project = make_project(tmp_path_factory, "hyol-gpu-test", api_port=18000, s3_port=19000)
    project.write_env()  # only the isolated ports; the setup questions fill in the rest
    yield project
    subprocess.run(["docker", "rm", "-f", "hyol-gpu-test-standalone"], capture_output=True)
    project.destroy()


def request(port: int, path: str, body: dict | None = None, api_key: str = "") -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(body).encode() if body is not None else None
    with urllib.request.urlopen(urllib.request.Request(f"http://localhost:{port}{path}", data, headers), timeout=120) as r:
        return json.load(r)


def chat(port: int, model: str, api_key: str = "") -> str:
    reply = request(port, "/v1/chat/completions", {
        "model": model, "messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 10,
    }, api_key)
    return reply["choices"][0]["message"]["content"]


def served_models(project) -> list[str]:
    return [m["id"] for m in request(project.api_port, "/v1/models", api_key=project.env()["VLLM_API_KEY"])["data"]]


def test_first_run_setup_and_serve(project):
    # Answers: preset 1, context 4096, default name, require an API key, write .env
    result = project.llm("serve", "--reconfigure", input="1\n4096\n\ny\ny\n", timeout=1200)
    assert result.returncode == 0, result.stdout + result.stderr

    env = project.env()
    assert env["MODEL"] == f"s3://models/{FIRST}"
    assert env["SERVED_MODEL_NAME"] == "qwen2.5-0.5b-instruct"
    assert env["S3_SECRET_KEY"] and env["VLLM_API_KEY"].startswith("sk-")
    assert env["API_PORT"] == "18000"  # settings the questions don't cover are kept
    assert "Serving qwen2.5-0.5b-instruct at http://localhost:18000/v1" in result.stdout


def test_api_requires_key(project):
    with pytest.raises(urllib.error.HTTPError) as error:
        chat(project.api_port, "qwen2.5-0.5b-instruct")
    assert error.value.code == 401
    assert chat(project.api_port, "qwen2.5-0.5b-instruct", project.env()["VLLM_API_KEY"]).strip()


def test_status_while_serving(project):
    result = project.llm("status")
    assert "s3    healthy" in result.stdout and "vllm  healthy" in result.stdout
    assert f"model {FIRST}" in result.stdout


def test_switch_model_while_running(project):
    result = project.llm("models", "use", SECOND, timeout=1200)
    assert result.returncode == 0, result.stdout + result.stderr

    assert served_models(project) == ["smollm2-135m-instruct"]
    assert chat(project.api_port, "smollm2-135m-instruct", project.env()["VLLM_API_KEY"]).strip()
    vllm_containers = project.compose("ps", "-q", "vllm").stdout.split()
    assert len(vllm_containers) == 1


def test_restart_keeps_models_and_asks_nothing(project):
    assert project.llm("down").returncode == 0
    assert project.containers() == []

    result = project.llm("serve", timeout=1200)  # no input: any question would fail it
    assert result.returncode == 0, result.stdout + result.stderr
    assert "already in the model store" in result.stdout
    assert served_models(project) == ["smollm2-135m-instruct"]


def test_image_runs_without_cli(project):
    """The image on its own: plain `docker run` with environment variables, no compose for vLLM."""
    project.compose("stop", "vllm")  # free the GPU
    env = project.env()
    subprocess.run([
        "docker", "run", "-d", "--name", "hyol-gpu-test-standalone", "--gpus", "all", "--ipc=host",
        "--network", f"{project.name}_default", "-p", f"{STANDALONE_PORT}:8000",
        "-e", f"MODEL=s3://models/{FIRST}", "-e", "MAX_MODEL_LEN=4096", "-e", "SERVED_MODEL_NAME=standalone",
        "-e", "AWS_ENDPOINT_URL=http://s3:9000", "-e", f"AWS_ACCESS_KEY_ID={env['S3_ACCESS_KEY']}",
        "-e", f"AWS_SECRET_ACCESS_KEY={env['S3_SECRET_KEY']}", "-e", "AWS_DEFAULT_REGION=us-east-1",
        "-e", "AWS_EC2_METADATA_DISABLED=true", "-e", "RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING=0",
        IMAGE,
    ], check=True, capture_output=True)

    deadline = time.monotonic() + 600
    while True:
        try:
            urllib.request.urlopen(f"http://localhost:{STANDALONE_PORT}/health", timeout=5)
            break
        except OSError:
            assert time.monotonic() < deadline, "the standalone container never became ready"
            time.sleep(5)
    assert chat(STANDALONE_PORT, "standalone").strip()
