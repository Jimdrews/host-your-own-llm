from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from host_your_own_llm.cli import app, deps
from host_your_own_llm.domain import StackStatus
from host_your_own_llm.errors import CliError

runner = CliRunner()


@pytest.fixture
def services(monkeypatch):
    services = Mock()
    monkeypatch.setattr(deps, "build_services", lambda: services)
    return services


@pytest.mark.parametrize("argv", [[], ["models"]])
def test_no_command_shows_help(argv):
    result = runner.invoke(app, argv)
    assert result.exit_code == 2
    assert "Usage:" in result.output and "Commands" in result.output


@pytest.mark.parametrize("command", ["add", "use", "rm"])
def test_missing_model_id(command):
    result = runner.invoke(app, ["models", command])
    assert result.exit_code == 2
    assert "Missing argument" in result.output


def test_serve(services):
    assert runner.invoke(app, ["serve", "--reconfigure"]).exit_code == 0
    services.stack.serve.assert_called_once_with(reconfigure=True)


def test_down(services):
    assert runner.invoke(app, ["down", "--purge"]).exit_code == 0
    services.stack.down.assert_called_once_with(purge=True)


def test_status(services):
    services.stack.status.return_value = StackStatus()
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Not configured yet" in result.output


def test_models_list(services):
    services.catalog.active_model_id.return_value = "org/b"
    services.catalog.list_models.return_value = ["org/a", "org/b"]
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert ["*", "org/b"] in [line.split() for line in result.output.splitlines()]


def test_models_list_empty(services):
    services.catalog.list_models.return_value = []
    assert "No models stored yet" in runner.invoke(app, ["models", "list"]).output


def test_models_use(services):
    assert runner.invoke(app, ["models", "use", "org/m", "--name", "x"]).exit_code == 0
    services.catalog.use.assert_called_once_with("org/m", served_name="x")


def test_models_rm(services):
    assert runner.invoke(app, ["models", "rm", "org/m", "-y"]).exit_code == 0
    services.catalog.remove.assert_called_once_with("org/m", confirmed=True)


def test_errors_are_short_messages(services):
    services.catalog.ensure_stored.side_effect = CliError("Not adding org/m.")
    result = runner.invoke(app, ["models", "add", "org/m"])
    assert result.exit_code == 1
    assert "Not adding org/m." in result.output
    assert "Traceback" not in result.output


def test_ctrl_c_cancels(services):
    services.stack.serve.side_effect = KeyboardInterrupt
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 1
    assert "Cancelled." in result.output


def test_outside_the_repository(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 1
    assert "inside the host-your-own-llm repository" in result.output
