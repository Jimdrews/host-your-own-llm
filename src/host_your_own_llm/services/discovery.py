"""Suggest popular models from the Hugging Face Hub that suit this GPU."""

from itertools import islice

from ..domain import FitVerdict, ModelSuggestion
from ..errors import CliError, ModelError
from ..ports import HubPort, VetterPort
from ..ui.console import console
from .vetting import assess_fit


class ModelDiscovery:
    def __init__(self, hub: HubPort, vetter: VetterPort):
        self.hub = hub
        self.vetter = vetter

    def suggest(
        self,
        query: str | None = None,
        trending: bool = False,
        limit: int = 20,
        include_too_big: bool = False,
    ) -> list[ModelSuggestion]:
        """Popular models, most popular first. Models too big for the GPU are left out unless asked for."""
        gpu = self.vetter.gpu()
        vram_gb = gpu.vram_gb if gpu else None
        try:
            with console.status("Searching Hugging Face..."):
                suggestions = (
                    ModelSuggestion(model=model, fit=assess_fit(model.size_gb, vram_gb))
                    for model in self.hub.search_models(query, trending)
                )
                kept = (s for s in suggestions if include_too_big or s.fit != FitVerdict.TOO_BIG)
                return list(islice(kept, limit))
        except ModelError as e:
            raise CliError(str(e)) from e
