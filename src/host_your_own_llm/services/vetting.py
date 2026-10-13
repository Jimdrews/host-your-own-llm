"""Decide whether a model can be served: it exists, we can access it, and it fits the GPU."""

from collections.abc import Callable

from ..adapters.gpu import detect_gpu
from ..adapters.huggingface import HubClient
from ..domain import FitVerdict, GpuInfo
from ..errors import ModelError
from ..ui import render
from ..ui.console import console
from ..ui.prompter import Prompter


def assess_fit(size_gb: float, vram_gb: float | None) -> FitVerdict:
    if not vram_gb:
        return FitVerdict.UNKNOWN
    if size_gb > vram_gb * 0.9:
        return FitVerdict.TOO_BIG
    if size_gb > vram_gb * 0.75:
        return FitVerdict.TIGHT
    return FitVerdict.FITS


class ModelVetter:
    def __init__(self, hub: HubClient, prompter: Prompter, gpu_probe: Callable[[], GpuInfo | None] = detect_gpu):
        self.hub = hub
        self.prompter = prompter
        self._gpu_probe = gpu_probe
        self._gpu: GpuInfo | None = None
        self._probed = False

    def gpu(self) -> GpuInfo | None:
        """Detect the GPU once, telling the user what was found."""
        if not self._probed:
            self._gpu = self._gpu_probe()
            self._probed = True
            render.print_gpu(self._gpu)
        return self._gpu

    def vet(self, model_id: str, token: str) -> str | None:
        """Check a model can be served. Return the token to use, or None to pick another."""
        gpu = self.gpu()
        with console.status(f"Checking {model_id}..."):
            try:
                check = self.hub.check_model(model_id, token)
            except ModelError as e:
                console.print(f"[red]{e}[/red]")
                return None

        if check.gated and not (token and self.hub.has_access(model_id, token)):
            console.print(f"[yellow]{model_id} is gated. Accept its license on huggingface.co, then paste a token.[/yellow]")
            entered = self.prompter.ask_secret("Hugging Face token")
            if entered is None:
                console.print("[red]No terminal to ask for a token; set HF_TOKEN in .env instead.[/red]")
                return None
            token = entered
            if not self.hub.has_access(model_id, token):
                console.print("[red]That token can't access this model.[/red]")
                return None

        console.print(f"{model_id}: [bold]{check.size_gb:.1f} GB[/bold] of weights")
        match assess_fit(check.size_gb, gpu.vram_gb if gpu else None):
            case FitVerdict.TOO_BIG:
                console.print(f"[red]That won't fit in {gpu.vram_gb:.0f} GB of VRAM.[/red]")
                if not self.prompter.confirm("Use it anyway?", default=False):
                    return None
            case FitVerdict.TIGHT:
                console.print("[yellow]That leaves little VRAM for context; consider a lower context length.[/yellow]")
        return token
