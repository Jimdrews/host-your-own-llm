import pytest

from host_your_own_llm.adapters.envfile import EnvFile, read_env, update_env
from host_your_own_llm.domain import StackConfig
from host_your_own_llm.errors import CliError


def test_read_env(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# comment\n\nMODEL=s3://models/a/b\nVLLM_API_KEY=abc==\n  SPACED = value \n")
    assert read_env(env) == {"MODEL": "s3://models/a/b", "VLLM_API_KEY": "abc==", "SPACED": "value"}


def test_update_env_keeps_other_lines(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# keep me\nA=1\nMODEL=old\n\nB=2\n")

    update_env(env, MODEL="new", NEW_KEY="x")

    assert env.read_text() == "# keep me\nA=1\nMODEL=new\n\nB=2\nNEW_KEY=x\n"


def test_update_env_creates_file(tmp_path):
    env = tmp_path / ".env"
    update_env(env, A="1")
    assert env.read_text() == "A=1\n"


def test_require_needs_a_file(tmp_path):
    with pytest.raises(CliError, match="Not configured yet"):
        EnvFile(tmp_path / ".env").require()


def test_load_without_file_gives_defaults(tmp_path):
    assert EnvFile(tmp_path / ".env").load() == StackConfig()


def test_save_keeps_keys_setup_does_not_own(tmp_path):
    path = tmp_path / ".env"
    path.write_text("# ports\nAPI_PORT=18000\nMODEL=old\n")
    env = EnvFile(path)

    env.save(StackConfig(MODEL="new"))

    values = read_env(path)
    assert values["API_PORT"] == "18000"
    assert values["MODEL"] == "new"
    assert path.read_text().startswith("# ports\nAPI_PORT=18000\nMODEL=new\n")
    assert env.load().api_port == 18000
