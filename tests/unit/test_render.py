import pytest

from host_your_own_llm.domain import FitVerdict, ModelSuggestion, ModelSummary, StackConfig, StackStatus
from host_your_own_llm.ui import render
from host_your_own_llm.ui.console import console


@pytest.mark.parametrize(
    ("platform", "curl", "body"),
    [
        ("win32", "curl.exe ", '{\\"model\\":\\"m\\"'),
        ("linux", "curl ", '{"model":"m"'),
    ],
)
def test_usage_command(platform, curl, body):
    command = render.usage_command(StackConfig(SERVED_MODEL_NAME="m"), platform)
    assert command.startswith(curl) and body in command
    assert "Authorization" not in command


def test_usage_command_with_api_key_and_port():
    config = StackConfig(SERVED_MODEL_NAME="m", VLLM_API_KEY="sk-1", API_PORT=18000)
    command = render.usage_command(config, "linux")
    assert "Authorization: Bearer sk-1" in command
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


@pytest.mark.parametrize(
    ("n", "text"), [(0, "0"), (999, "999"), (1790, "1.8k"), (30604529, "30.6M"), (2_500_000_000, "2.5B")]
)
def test_compact_count(n, text):
    assert render.compact_count(n) == text


def test_suggestions_table(capsys):
    gb = 1024**3
    suggestions = [
        ModelSuggestion(
            model=ModelSummary(model_id="org/a", downloads=1500, likes=12, size_bytes=gb), fit=FitVerdict.FITS
        ),
        ModelSuggestion(model=ModelSummary(model_id="org/b", gated=True, size_bytes=8 * gb), fit=FitVerdict.TIGHT),
        ModelSuggestion(model=ModelSummary(model_id="org/c", size_bytes=20 * gb), fit=FitVerdict.TOO_BIG),
    ]
    console.print(render.suggestions_table(suggestions), width=200)
    rows = {line.split()[1]: line for line in capsys.readouterr().out.splitlines()[1:] if line.strip()}
    assert "1.0 GB" in rows["org/a"] and "1.5k" in rows["org/a"]
    assert "tight fit, gated" in rows["org/b"]
    assert "too big" in rows["org/c"]
