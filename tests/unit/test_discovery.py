import pytest

from fakes import FakeHub, FakePrompter
from host_your_own_llm.domain import FitVerdict, GpuInfo, ModelSummary
from host_your_own_llm.errors import CliError, ModelError
from host_your_own_llm.services.discovery import ModelDiscovery
from host_your_own_llm.services.vetting import ModelVetter

GB = 1024**3
LISTING = [
    ModelSummary(model_id="org/small", downloads=300, size_bytes=1 * GB),
    ModelSummary(model_id="org/big", downloads=200, size_bytes=20 * GB),
    ModelSummary(model_id="org/tight", downloads=100, size_bytes=8 * GB),
]


def discovery(hub, gpu=GpuInfo(name="Test GPU", vram_gb=10)):
    return ModelDiscovery(hub, ModelVetter(hub, FakePrompter(), gpu_probe=lambda: gpu))


def ids(suggestions):
    return [s.model.model_id for s in suggestions]


def test_too_big_models_are_hidden():
    suggestions = discovery(FakeHub(listing=LISTING)).suggest()
    assert ids(suggestions) == ["org/small", "org/tight"]
    assert [s.fit for s in suggestions] == [FitVerdict.FITS, FitVerdict.TIGHT]


def test_too_big_models_can_be_included():
    suggestions = discovery(FakeHub(listing=LISTING)).suggest(include_too_big=True)
    assert ids(suggestions) == ["org/small", "org/big", "org/tight"]
    assert suggestions[1].fit == FitVerdict.TOO_BIG


def test_without_a_gpu_nothing_is_hidden():
    suggestions = discovery(FakeHub(listing=LISTING), gpu=None).suggest()
    assert ids(suggestions) == ["org/small", "org/big", "org/tight"]
    assert {s.fit for s in suggestions} == {FitVerdict.UNKNOWN}


def test_limit_counts_models_shown():
    assert ids(discovery(FakeHub(listing=LISTING)).suggest(limit=1)) == ["org/small"]
    assert ids(discovery(FakeHub(listing=LISTING)).suggest(limit=2)) == ["org/small", "org/tight"]


def test_query_and_sort_are_passed_on():
    hub = FakeHub(listing=LISTING)
    assert ids(discovery(hub).suggest("tight", trending=True)) == ["org/tight"]
    assert hub.searches == [("tight", True)]


def test_hub_errors_are_reported():
    with pytest.raises(CliError, match="Couldn't reach"):
        discovery(FakeHub(listing=ModelError("Couldn't reach Hugging Face: offline"))).suggest()
