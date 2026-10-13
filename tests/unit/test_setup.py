from host_your_own_llm.adapters.envfile import EnvFile, read_env
from host_your_own_llm.domain import ModelSummary
from host_your_own_llm.errors import ModelError
from host_your_own_llm.services.discovery import ModelDiscovery
from host_your_own_llm.services.setup import SetupWizard

from fakes import FakeHub, FakePrompter, FakeVetter

POPULAR = [
    ModelSummary(model_id="org/popular", size_bytes=1024**3),
    ModelSummary(model_id="org/second", size_bytes=1024**3),
]


def wizard(path, *answers, vetter=None, listing=POPULAR):
    vetter = vetter or FakeVetter(token=None)
    discovery = ModelDiscovery(FakeHub(listing=listing), vetter)
    return SetupWizard(EnvFile(path), vetter, FakePrompter(*answers), discovery)


def test_first_run_writes_env(tmp_path):
    path = tmp_path / ".env"
    path.write_text("API_PORT=18000\n")
    # preset 1, context 4096, default name, require an API key, write .env
    assert wizard(path, 1, 4096, "", True, True).configure()

    env = read_env(path)
    assert env["MODEL"] == "s3://models/Qwen/Qwen2.5-0.5B-Instruct"
    assert env["SERVED_MODEL_NAME"] == "qwen2.5-0.5b-instruct"
    assert env["MAX_MODEL_LEN"] == "4096"
    assert env["VLLM_API_KEY"].startswith("sk-")
    assert env["S3_SECRET_KEY"] and env["S3_ACCESS_KEY"] == "modelstore"
    assert env["API_PORT"] == "18000"


def test_reconfigure_keeps_secrets(tmp_path):
    path = tmp_path / ".env"
    path.write_text("S3_SECRET_KEY=keep\nVLLM_API_KEY=sk-keep\nMAX_MODEL_LEN=2048\n")
    # preset 2, default context, default name, keep the API key, write .env
    assert wizard(path, 2, "", "", "", True).configure()

    env = read_env(path)
    assert env["S3_SECRET_KEY"] == "keep"
    assert env["VLLM_API_KEY"] == "sk-keep"
    assert env["MAX_MODEL_LEN"] == "2048"


def test_declining_writes_nothing(tmp_path):
    path = tmp_path / ".env"
    assert not wizard(path, 1, "", "", False, False).configure()
    assert not path.exists()


def test_other_model_after_bad_choice(tmp_path, capsys):
    vetter = FakeVetter()
    assert wizard(tmp_path / ".env", 99, 6, "org/custom", "", "", False, True, vetter=vetter).configure()
    assert "Pick one of the numbers" in capsys.readouterr().out
    assert vetter.vetted == ["org/custom"]
    assert read_env(tmp_path / ".env")["SERVED_MODEL_NAME"] == "custom"


def test_rejected_model_asks_again(tmp_path):
    vetter = FakeVetter()
    vetter.vet = lambda model_id, token: None if model_id.endswith("0.5B-Instruct") else "hf"
    assert wizard(tmp_path / ".env", 1, 2, "", "", False, True, vetter=vetter).configure()
    env = read_env(tmp_path / ".env")
    assert env["MODEL"] == "s3://models/Qwen/Qwen2.5-1.5B-Instruct"
    assert env["HF_TOKEN"] == "hf"


def test_browse_picks_a_popular_model(tmp_path):
    vetter = FakeVetter()
    # browse, blank search, pick #2, then default context, default name, no API key, write .env
    assert wizard(tmp_path / ".env", 7, "", 2, "", "", False, True, vetter=vetter).configure()
    assert vetter.vetted == ["org/second"]
    assert read_env(tmp_path / ".env")["MODEL"] == "s3://models/org/second"


def test_browse_can_go_back(tmp_path, capsys):
    # browse, search "zzz" (nothing), browse again, go back with 0, then preset 1
    assert wizard(tmp_path / ".env", 7, "zzz", 7, "", 0, 1, "", "", False, True).configure()
    assert "No matching models" in capsys.readouterr().out
    assert read_env(tmp_path / ".env")["MODEL"] == "s3://models/Qwen/Qwen2.5-0.5B-Instruct"


def test_browse_offline_goes_back(tmp_path, capsys):
    path = tmp_path / ".env"
    assert wizard(
        path, 7, "", 1, "", "", False, True, listing=ModelError("Couldn't reach Hugging Face: offline")
    ).configure()
    assert "Couldn't reach Hugging Face" in capsys.readouterr().out
    assert read_env(path)["MODEL"] == "s3://models/Qwen/Qwen2.5-0.5B-Instruct"
