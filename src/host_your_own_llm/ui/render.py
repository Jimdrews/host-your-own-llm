"""How things look in the terminal. Builders return strings or tables; print_* functions print them."""

import sys
from collections.abc import Iterable

from rich.table import Table

from ..domain import SERVICES, FitVerdict, GpuInfo, ModelSuggestion, Preset, StackConfig, StackStatus
from .console import console


def usage_command(config: StackConfig, platform: str | None = None) -> str:
    """A curl command that sends one chat request to the served model."""
    body = f'{{"model":"{config.display_name}","messages":[{{"role":"user","content":"Hello!"}}]}}'
    auth = f' -H "Authorization: Bearer {config.vllm_api_key}"' if config.vllm_api_key else ""
    curl_exe = "curl"
    if (platform or sys.platform) == "win32":
        # PowerShell strips unescaped double quotes from arguments to native programs
        curl_exe = "curl.exe"
        body = body.replace('"', '\\"')
    return f"{curl_exe} {config.api_url}/v1/chat/completions -H \"Content-Type: application/json\"{auth} -d '{body}'"


def print_usage(config: StackConfig) -> None:
    console.print(f"[green]Serving [bold]{config.display_name}[/bold] at {config.api_url}/v1[/green]")
    console.print(usage_command(config), markup=False, highlight=False, soft_wrap=True)


def print_status(status: StackStatus) -> None:
    for service in SERVICES:
        console.print(f"{service:5} {status.state(service)}")
    if status.config is None:
        console.print("Not configured yet. Run `uv run llm serve`.")
        return
    console.print(f"model {status.config.active_model_id}")
    if status.api_up:
        print_usage(status.config)


def print_gpu(gpu: GpuInfo | None) -> None:
    if gpu is None:
        console.print("[yellow]No NVIDIA GPU detected; skipping VRAM checks.[/yellow]")
    else:
        console.print(f"GPU: [bold]{gpu.name}[/bold] ({gpu.vram_gb:.0f} GB)")


def models_table(stored: Iterable[str], active: str) -> Table:
    table = Table(show_header=False, box=None)
    for model_id in stored:
        table.add_row("[green]*[/green]" if model_id == active else " ", model_id)
    return table


def presets_table(presets: Iterable[Preset]) -> Table:
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("Model")
    table.add_column("Notes")
    count = 0
    for count, preset in enumerate(presets, 1):
        table.add_row(str(count), preset.model_id, preset.note)
    table.add_row(str(count + 1), "Other", "Enter any Hugging Face model id")
    table.add_row(str(count + 2), "Browse", "Search popular models on Hugging Face")
    return table


def compact_count(n: int) -> str:
    """30604529 -> 30.6M, 1790 -> 1.8k."""
    for size, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "k")):
        if n >= size:
            return f"{n / size:.1f}{suffix}"
    return str(n)


FIT_NOTES = {FitVerdict.TIGHT: "[yellow]tight fit[/yellow]", FitVerdict.TOO_BIG: "[red]too big[/red]"}


def suggestions_table(suggestions: Iterable[ModelSuggestion]) -> Table:
    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("#", justify="right")
    table.add_column("Model")
    table.add_column("Size", justify="right")
    table.add_column("Downloads", justify="right")
    table.add_column("Likes", justify="right")
    table.add_column("Notes")
    for i, suggestion in enumerate(suggestions, 1):
        model = suggestion.model
        notes = [note for note in (FIT_NOTES.get(suggestion.fit), "gated" if model.gated else None) if note]
        table.add_row(
            str(i),
            model.model_id,
            f"{model.size_gb:.1f} GB",
            compact_count(model.downloads),
            compact_count(model.likes),
            ", ".join(notes),
        )
    return table


def summary_table(config: StackConfig) -> Table:
    values = config.to_env()
    table = Table(show_header=False)
    for key in ("MODEL", "SERVED_MODEL_NAME", "MAX_MODEL_LEN", "VLLM_API_KEY"):
        table.add_row(key, values[key] or "[dim](none)[/dim]")
    return table
