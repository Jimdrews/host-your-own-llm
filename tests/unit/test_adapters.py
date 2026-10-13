import json
import subprocess

import pytest
from huggingface_hub.errors import GatedRepoError

from fakes import FakeCompose
from host_your_own_llm.adapters import gpu, huggingface
from host_your_own_llm.adapters.compose import Compose
from host_your_own_llm.adapters.store import ModelStore
from host_your_own_llm.errors import CliError, ModelError


def completed(stdout="", returncode=0, stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


class Runner:
    def __init__(self, *results):
        self.results = list(results)
        self.commands = []

    def __call__(self, cmd, **kwargs):
        self.commands.append(cmd)
        return self.results.pop(0)


def test_compose_returns_quiet_output(tmp_path):
    runner = Runner(completed("out"))
    assert Compose(tmp_path, runner).run("ps", quiet=True) == "out"
    assert runner.commands[0] == ["docker", "compose", "--progress", "quiet", "ps"]


def test_compose_failure_shows_output(tmp_path, capsys):
    with pytest.raises(CliError, match="`docker compose up` failed"):
        Compose(tmp_path, Runner(completed("boom", returncode=1))).run("up", quiet=True)
    assert "boom" in capsys.readouterr().out


def test_compose_ps(tmp_path):
    lines = [json.dumps({"Service": "s3", "State": "running", "Health": "healthy"}),
             json.dumps({"Service": "vllm", "State": "exited", "Health": ""})]
    states = Compose(tmp_path, Runner(completed("\n".join(lines) + "\n"))).ps()
    assert [(s.service, s.effective) for s in states] == [("s3", "healthy"), ("vllm", "exited")]


def test_store_runs_tool_container():
    compose = FakeCompose()
    ModelStore(compose).seed("org/m")
    call = compose.calls[0]
    assert call[:4] == ("run", "run", "--rm", "-T")
    assert call[-3:] == ("store", "seed", "org/m")


def test_store_removes_container_on_ctrl_c():
    """Faked because a real Ctrl+C can't be sent reliably to a child process on Windows."""
    compose = FakeCompose()

    def interrupted(*args, quiet=False):
        raise KeyboardInterrupt

    compose.run = interrupted
    with pytest.raises(KeyboardInterrupt):
        ModelStore(compose).seed()
    assert compose.calls[0][0] == "rm"
    assert compose.calls[0][1].startswith("host-your-own-llm-store-")


def test_gated_error_is_not_reported_as_not_found(monkeypatch):
    """GatedRepoError subclasses RepositoryNotFoundError, so it must be caught first.

    Real Hugging Face lookups return gated models with a flag instead of raising, so the
    functional tests can't reach this branch; the error is built by hand here.
    """
    error = GatedRepoError.__new__(GatedRepoError)  # the real constructor needs an HTTP response
    Exception.__init__(error, "gated")

    def raise_gated(*args, **kwargs):
        raise error

    monkeypatch.setattr(huggingface, "model_info", raise_gated)
    with pytest.raises(ModelError, match="is gated"):
        huggingface.HubClient().check_model("org/model", None)


def test_detect_gpu():
    found = gpu.detect_gpu(Runner(completed("NVIDIA GeForce RTX 4070, 12282\n")))
    assert found.name == "NVIDIA GeForce RTX 4070"
    assert round(found.vram_gb) == 12


def test_no_gpu():
    def missing(cmd, **kwargs):
        raise FileNotFoundError

    assert gpu.detect_gpu(missing) is None
