import pytest

from host_your_own_llm.domain import StackConfig, StackStatus
from host_your_own_llm.ui import render
from host_your_own_llm.ui.console import console


@pytest.mark.parametrize(("platform", "curl", "body"), [
    ("win32", "curl.exe ", '{\\"model\\":\\"m\\"'),
    ("linux", "curl ", '{"model":"m"'),
])
def test_usage_command(platform, curl, body):
    command = render.usage_command(StackConfig(SERVED_MODEL_NAME="m"), platform)
    assert command.startswith(curl) and body in command
    assert "Authorization" not in command


def test_usage_command_with_api_key_and_port():
    config = StackConfig(SERVED_MODEL_NAME="m", VLLM_API_KEY="sk-1", API_PORT=18000)
    command = render.usage_command(config, "linux")
    assert 'Authorization: Bearer sk-1' in command
    assert "http://localhost:18000/v1/chat/completions" in command


def test_status_when_not_configured(capsys):
    render.print_status(StackStatus(services={"s3": "healthy"}))
    out = capsys.readouterr().out
    assert "s3    healthy" in out and "vllm  stopped" in out
    assert "Not configured yet" in out


def test_status_while_serving(capsys):
    config = StackConfig(MODEL="s3://models/org/m", SERVED_MODEL_NAME="m")
    render.print_status(StackStatus(services={"s3": "healthy", "vllm": "healthy"}, config=config, api_up=True))
    out = capsys.readouterr().out
    assert "model org/m" in out
    assert "Serving m at http://localhost:8000/v1" in out


def test_models_table_marks_active(capsys):
    console.print(render.models_table(["a/one", "b/two"], active="b/two"))
    lines = [line.split() for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert lines == [["a/one"], ["*", "b/two"]]
