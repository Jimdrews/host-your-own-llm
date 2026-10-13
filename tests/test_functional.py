"""The real `llm` CLI against a real model store and real Hugging Face, in an isolated project.

The tests run in order and share one project, the way a person would use the tool.
No GPU is needed: vLLM itself is never started here (see test_gpu.py for that).
"""

import time

import pytest

from conftest import make_project
from host_your_own_llm.adapters.gpu import detect_gpu

pytestmark = pytest.mark.functional

LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"  # 4 MB
PHI = "hf-internal-testing/tiny-random-PhiForCausalLM"  # 0.3 MB


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    project = make_project(tmp_path_factory, "hyol-test", api_port=18000, s3_port=19000)
    project.write_env(S3_ACCESS_KEY="testkey", S3_SECRET_KEY="test-secret-123", S3_BUCKET="models",
                      MODEL=f"s3://models/{LLAMA}", SERVED_MODEL_NAME="tiny-llama", HF_TOKEN="")
    yield project
    project.destroy()


def stored(project) -> list[str]:
    result = project.llm("models", "list")
    assert result.returncode == 0, result.stdout + result.stderr
    if "No models stored yet" in result.stdout:
        return []
    return [line.strip().lstrip("*").strip() for line in result.stdout.splitlines() if line.strip()]


@pytest.mark.parametrize("argv", [[], ["models"]])
def test_missing_command_shows_help(project, argv):
    result = project.llm(*argv)
    assert result.returncode == 2
    output = result.stdout + result.stderr
    assert f"Usage: llm {' '.join(argv)}".strip() in output
    assert "Commands" in output


@pytest.mark.parametrize("command", ["add", "use", "rm"])
def test_missing_model_id(project, command):
    result = project.llm("models", command)
    assert result.returncode == 2
    assert f"Usage: llm models {command}" in result.stderr
    assert "Missing argument 'MODEL_ID'" in result.stderr


def test_search_lists_popular_models(project):
    result = project.llm("models", "search", "qwen", "--limit", "3", "--all", timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("Qwen/") == 3
    assert "models add" in result.stdout


def test_add_copies_model_into_store(project):
    result = project.llm("models", "add", LLAMA)
    assert result.returncode == 0, result.stdout + result.stderr

    keys = project.bucket_keys(f"{LLAMA}/")
    assert f"{LLAMA}/model.safetensors" in keys
    assert f"{LLAMA}/config.json" in keys
    assert f"{LLAMA}/.complete" in keys
    assert stored(project) == [LLAMA]


def test_add_again_skips_download(project):
    start = time.monotonic()
    result = project.llm("models", "add", LLAMA)
    assert result.returncode == 0
    assert "already in the model store" in result.stdout
    assert "Downloading" not in result.stdout
    assert time.monotonic() - start < 60


def test_interrupted_copy_is_completed(project):
    """An interrupted copy leaves weights without the marker; it must not count as stored."""
    project.s3_client().put_object(Bucket="models", Key=f"{PHI}/model.safetensors", Body=b"partial")
    assert PHI not in stored(project)

    result = project.llm("models", "add", PHI)
    assert result.returncode == 0, result.stdout + result.stderr

    keys = project.bucket_keys(f"{PHI}/")
    assert f"{PHI}/.complete" in keys and f"{PHI}/config.json" in keys
    weights = project.s3_client().get_object(Bucket="models", Key=f"{PHI}/model.safetensors")["Body"].read()
    assert weights != b"partial"
    assert stored(project) == sorted([LLAMA, PHI])


@pytest.mark.parametrize(("model_id", "message"), [
    ("nobody-here/definitely-not-a-model", "not found"),
    ("hf-internal-testing/tiny-random-GPT2LMHeadModel", "no .safetensors"),
])
def test_unusable_models_are_rejected(project, model_id, message):
    result = project.llm("models", "add", model_id)
    assert result.returncode == 1
    assert message in result.stdout
    assert project.bucket_keys(f"{model_id}/") == []


def test_gated_model_needs_a_token(project):
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    result = project.llm("models", "add", model_id, input="\n", timeout=120)
    assert result.returncode == 1
    assert "is gated" in result.stdout
    assert "set HF_TOKEN in .env" in result.stdout  # no terminal, so it must not wait for a token
    assert project.bucket_keys(f"{model_id}/") == []


@pytest.mark.skipif(detect_gpu() is None, reason="needs a GPU to compare the model size against")
def test_model_too_big_for_gpu_can_be_declined(project):
    model_id = "Qwen/Qwen2.5-32B-Instruct"  # 65 GB of weights
    result = project.llm("models", "add", model_id, input="n\n")
    assert result.returncode == 1
    assert "won't fit" in result.stdout
    assert project.bucket_keys(f"{model_id}/") == []


def test_use_switches_model_while_stopped(project):
    result = project.llm("models", "use", PHI, "--name", "phi")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Now using" in result.stdout

    env = project.env()
    assert env["MODEL"] == f"s3://models/{PHI}"
    assert env["SERVED_MODEL_NAME"] == "phi"
    assert env["API_PORT"] == "18000"  # other settings are kept


def test_rm_refuses_active_model(project):
    result = project.llm("models", "rm", PHI, "-y")
    assert result.returncode == 1
    assert "active model" in result.stdout
    assert f"{PHI}/.complete" in project.bucket_keys(f"{PHI}/")


def test_rm_can_be_declined(project):
    result = project.llm("models", "rm", LLAMA, input="n\n")
    assert result.returncode == 1
    assert LLAMA in stored(project)


def test_rm_removes_model(project):
    result = project.llm("models", "rm", LLAMA, "-y")
    assert result.returncode == 0, result.stdout + result.stderr
    assert project.bucket_keys(f"{LLAMA}/") == []
    assert stored(project) == [PHI]


def test_use_adds_missing_model(project):
    result = project.llm("models", "use", LLAMA)
    assert result.returncode == 0, result.stdout + result.stderr
    assert stored(project) == sorted([LLAMA, PHI])
    assert project.env()["SERVED_MODEL_NAME"] == "tiny-random-llamaforcausallm"


def test_status(project):
    result = project.llm("status")
    assert result.returncode == 0
    assert "s3    healthy" in result.stdout
    assert "vllm  stopped" in result.stdout
    assert f"model {LLAMA}" in result.stdout


def test_down_keeps_models(project):
    result = project.llm("down")
    assert result.returncode == 0
    assert project.containers() == []
    assert project.volume_exists()
    assert stored(project) == sorted([LLAMA, PHI])  # starts the store again


def test_purge_can_be_declined(project):
    result = project.llm("down", "--purge", input="n\n")
    assert result.returncode == 1
    assert project.volume_exists()


def test_purge_deletes_models(project):
    result = project.llm("down", "--purge", input="y\n")
    assert result.returncode == 0
    assert project.containers() == []
    assert not project.volume_exists()
    assert stored(project) == []
