import pytest

from host_your_own_llm.adapters.envfile import EnvFile
from host_your_own_llm.domain import ServiceState
from host_your_own_llm.errors import CliError
from host_your_own_llm.services.stack import StackService

from fakes import FakeClock, FakeCompose, FakePrompter, FakeStore


@pytest.fixture
def env(tmp_path):
    path = tmp_path / ".env"
    path.write_text("MODEL=s3://models/org/m\nSERVED_MODEL_NAME=m\nAPI_PORT=18000\n")
    return EnvFile(path)


def service(
    env, compose=None, store=None, prompter=None, configure=lambda: True, healthy=True, nvidia=True, clock=None
):
    clock = clock or FakeClock()
    return StackService(
        compose or FakeCompose(),
        store or FakeStore(),
        env,
        prompter or FakePrompter(),
        configure,
        docker_check=lambda: nvidia,
        health=lambda url: healthy,
        clock=clock,
        sleep=clock.sleep,
        timeout=30,
    )


def test_serve_builds_seeds_and_starts(env, capsys):
    compose, store = FakeCompose(), FakeStore()
    service(env, compose, store).serve()
    assert compose.calls == [("build", "vllm"), ("up", "vllm")]
    assert store.calls == [("start",), ("seed", None)]
    assert "Serving m at http://localhost:18000/v1" in capsys.readouterr().out


def test_serve_asks_setup_questions_on_first_run(tmp_path):
    asked = []
    service(EnvFile(tmp_path / ".env"), configure=lambda: asked.append(1) or True).serve()
    assert asked == [1]


def test_serve_only_reconfigures_when_asked(env):
    asked = []
    service(env, configure=lambda: asked.append(1) or True).serve()
    assert asked == []
    service(env, configure=lambda: asked.append(1) or True).serve(reconfigure=True)
    assert asked == [1]


def test_serve_cancelled_setup(tmp_path):
    compose = FakeCompose()
    with pytest.raises(CliError, match="Setup cancelled"):
        service(EnvFile(tmp_path / ".env"), compose, configure=lambda: False).serve()
    assert compose.calls == []


def test_serve_warns_without_nvidia_runtime(env, capsys):
    service(env, nvidia=False).serve()
    assert "NVIDIA runtime" in capsys.readouterr().out


def test_start_vllm_times_out_with_logs(env):
    compose = FakeCompose()
    with pytest.raises(CliError, match="didn't become ready"):
        service(env, compose, healthy=False).start_vllm(env.load())
    assert ("logs", "vllm") in compose.calls


def test_down_keeps_models(env):
    compose = FakeCompose()
    service(env, compose).down()
    assert compose.calls == [("down", False)]


def test_purge_can_be_declined(env):
    compose = FakeCompose()
    with pytest.raises(CliError, match="Nothing deleted"):
        service(env, compose, prompter=FakePrompter(False)).down(purge=True)
    assert compose.calls == []


def test_purge_deletes_volume(env):
    compose = FakeCompose()
    service(env, compose, prompter=FakePrompter(True)).down(purge=True)
    assert compose.calls == [("down", True)]


def test_status_asks_the_api(env):
    compose = FakeCompose(
        ps=[
            ServiceState(Service="s3", State="running", Health="healthy"),
            ServiceState(Service="vllm", State="running", Health="starting"),
        ]
    )
    status = service(env, compose, healthy=True).status()
    assert status.state("s3") == "healthy"
    assert status.state("vllm") == "healthy"
    assert status.api_up
    assert status.config is not None and status.config.active_model_id == "org/m"


def test_status_when_stopped_and_unconfigured(tmp_path):
    status = service(EnvFile(tmp_path / ".env"), healthy=True).status()
    assert status.state("vllm") == "stopped"
    assert status.config is None and not status.api_up
