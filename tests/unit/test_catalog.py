import pytest

from fakes import FakePrompter, FakeStack, FakeStore, FakeVetter
from host_your_own_llm.adapters.envfile import EnvFile, read_env
from host_your_own_llm.errors import CliError
from host_your_own_llm.services.catalog import ModelCatalog


@pytest.fixture
def env(tmp_path):
    path = tmp_path / ".env"
    path.write_text("MODEL=s3://models/org/active\nSERVED_MODEL_NAME=active\nHF_TOKEN=hf_old\nAPI_PORT=18000\n")
    return EnvFile(path)


def catalog(env, store=None, vetter=None, prompter=None, stack=None):
    return ModelCatalog(env, store or FakeStore(), vetter or FakeVetter(token=None), prompter or FakePrompter(), stack or FakeStack())


def test_needs_configuration(tmp_path):
    with pytest.raises(CliError, match="Not configured yet"):
        catalog(EnvFile(tmp_path / ".env")).list_models()


def test_add_skips_stored_model(env, capsys):
    store, vetter = FakeStore(["org/m"]), FakeVetter()
    catalog(env, store, vetter).ensure_stored("org/m")
    assert ("seed", "org/m") not in store.calls
    assert vetter.vetted == []
    assert "already in the model store" in capsys.readouterr().out


def test_add_copies_new_model(env):
    store = FakeStore()
    catalog(env, store).ensure_stored("org/m")
    assert store.calls == [("start",), ("seed", "org/m")]
    assert read_env(env.path)["HF_TOKEN"] == "hf_old"


def test_add_saves_new_token(env):
    catalog(env, vetter=FakeVetter(token="hf_new")).ensure_stored("org/m")
    assert read_env(env.path)["HF_TOKEN"] == "hf_new"


def test_add_rejected_model(env):
    store = FakeStore()
    with pytest.raises(CliError, match="Not adding org/m"):
        catalog(env, store, FakeVetter(reject=True)).ensure_stored("org/m")
    assert store.models == []


def test_use_while_stopped(env, capsys):
    stack = FakeStack(running=False)
    catalog(env, stack=stack).use("org/new")
    values = read_env(env.path)
    assert values["MODEL"] == "s3://models/org/new"
    assert values["SERVED_MODEL_NAME"] == "new"
    assert values["API_PORT"] == "18000"
    assert stack.started == []
    assert "Now using org/new" in capsys.readouterr().out


def test_use_while_running_restarts_vllm(env):
    stack = FakeStack(running=True)
    catalog(env, stack=stack).use("org/new", served_name="custom")
    assert [c.served_model_name for c in stack.started] == ["custom"]


def test_rm_refuses_active_model(env):
    store = FakeStore(["org/active"])
    with pytest.raises(CliError, match="active model"):
        catalog(env, store).remove("org/active", confirmed=True)
    assert store.models == ["org/active"]


def test_rm_can_be_declined(env):
    store = FakeStore(["org/old"])
    with pytest.raises(CliError, match="Nothing deleted"):
        catalog(env, store, prompter=FakePrompter(False)).remove("org/old")
    assert store.models == ["org/old"]


def test_rm_confirmed(env):
    store = FakeStore(["org/old"])
    catalog(env, store, prompter=FakePrompter(True)).remove("org/old")
    assert store.models == []
