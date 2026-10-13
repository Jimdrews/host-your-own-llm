import pytest

from host_your_own_llm.domain import FitVerdict, GpuInfo, ModelCheck
from host_your_own_llm.errors import ModelError
from host_your_own_llm.services.vetting import ModelVetter, assess_fit

from fakes import FakeHub, FakePrompter

GB = 1024**3
GPU = GpuInfo(name="Test GPU", vram_gb=10)


@pytest.mark.parametrize(
    ("size_gb", "vram_gb", "verdict"),
    [
        (5, None, FitVerdict.UNKNOWN),
        (5, 10, FitVerdict.FITS),
        (7.5, 10, FitVerdict.FITS),
        (8, 10, FitVerdict.TIGHT),
        (9, 10, FitVerdict.TIGHT),
        (9.5, 10, FitVerdict.TOO_BIG),
    ],
)
def test_assess_fit(size_gb, vram_gb, verdict):
    assert assess_fit(size_gb, vram_gb) == verdict


def vetter(hub, prompter=None, gpu=GPU):
    return ModelVetter(hub, prompter or FakePrompter(), gpu_probe=lambda: gpu)


def test_unusable_model(capsys):
    hub = FakeHub({"org/m": ModelError("org/m was not found")})
    assert vetter(hub).vet("org/m", "") is None
    assert "not found" in capsys.readouterr().out


def test_open_model_keeps_token(capsys):
    hub = FakeHub({"org/m": ModelCheck(model_id="org/m", size_bytes=2 * GB, gated=False)})
    assert vetter(hub).vet("org/m", "hf_old") == "hf_old"
    assert "2.0 GB" in capsys.readouterr().out


def test_gated_model_without_terminal(capsys):
    hub = FakeHub({"org/m": ModelCheck(model_id="org/m", size_bytes=GB, gated=True)})
    assert vetter(hub, FakePrompter(tty=False)).vet("org/m", "") is None
    assert "set HF_TOKEN in .env" in capsys.readouterr().out


def test_gated_model_asks_for_token():
    hub = FakeHub({"org/m": ModelCheck(model_id="org/m", size_bytes=GB, gated=True)}, tokens={"org/m": {"hf_good"}})
    assert vetter(hub, FakePrompter("hf_good")).vet("org/m", "") == "hf_good"
    assert vetter(hub, FakePrompter("hf_bad")).vet("org/m", "") is None


def test_gated_model_with_working_token_asks_nothing():
    hub = FakeHub({"org/m": ModelCheck(model_id="org/m", size_bytes=GB, gated=True)}, tokens={"org/m": {"hf_good"}})
    assert vetter(hub, FakePrompter()).vet("org/m", "hf_good") == "hf_good"


@pytest.mark.parametrize(("answer", "expected"), [(False, None), (True, "")])
def test_too_big_asks_to_continue(capsys, answer, expected):
    hub = FakeHub({"org/big": ModelCheck(model_id="org/big", size_bytes=20 * GB, gated=False)})
    assert vetter(hub, FakePrompter(answer)).vet("org/big", "") == expected
    assert "won't fit in 10 GB" in capsys.readouterr().out


def test_tight_fit_warns(capsys):
    hub = FakeHub({"org/m": ModelCheck(model_id="org/m", size_bytes=8 * GB, gated=False)})
    assert vetter(hub).vet("org/m", "") == ""
    assert "little VRAM" in capsys.readouterr().out


def test_gpu_is_detected_once(capsys):
    probes = []
    hub = FakeHub({"org/m": ModelCheck(model_id="org/m", size_bytes=GB, gated=False)})
    v = ModelVetter(hub, FakePrompter(), gpu_probe=lambda: probes.append(1))
    v.vet("org/m", "")
    v.vet("org/m", "")
    assert probes == [1]
    assert "No NVIDIA GPU detected" in capsys.readouterr().out
