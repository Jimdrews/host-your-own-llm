"""`models list|search|add|use|rm`."""

from typing import Annotated

import typer

from ..ui import render
from ..ui.console import console
from . import deps
from .errors import reports_errors

app = typer.Typer(no_args_is_help=True, help="Find, list, add, switch or remove stored models.")

HfModelId = Annotated[str, typer.Argument(metavar="MODEL_ID", help="Hugging Face id, e.g. Qwen/Qwen2.5-1.5B-Instruct")]


@app.command("list")
@reports_errors
def list_models() -> None:
    """List stored models."""
    catalog = deps.build_services().catalog
    active = catalog.active_model_id()
    stored = catalog.list_models()
    if not stored:
        console.print("No models stored yet. Add one with `uv run llm models add <id>`.")
    console.print(render.models_table(stored, active))


@app.command()
@reports_errors
def search(
    query: Annotated[str | None, typer.Argument(metavar="QUERY", help="filter by name, e.g. qwen")] = None,
    trending: Annotated[bool, typer.Option("--trending", help="sort by what's trending instead of downloads")] = False,
    limit: Annotated[int, typer.Option(min=1, help="how many models to show")] = 20,
    show_all: Annotated[bool, typer.Option("--all", help="include models too big for your GPU")] = False,
) -> None:
    """Find popular models on Hugging Face that fit your GPU."""
    suggestions = deps.build_services().discovery.suggest(query, trending=trending, limit=limit, include_too_big=show_all)
    if not suggestions:
        console.print("No matching models.")
        return
    console.print(render.suggestions_table(suggestions))
    console.print("Add one with `uv run llm models add <id>`.")


@app.command()
@reports_errors
def add(model_id: HfModelId) -> None:
    """Copy a model from Hugging Face into the store."""
    deps.build_services().catalog.ensure_stored(model_id)


@app.command()
@reports_errors
def use(
    model_id: HfModelId,
    name: Annotated[str | None, typer.Option(help="model name for API requests (default: derived from the id)")] = None,
) -> None:
    """Switch the served model, adding it first if needed."""
    deps.build_services().catalog.use(model_id, served_name=name)


@app.command()
@reports_errors
def rm(
    model_id: Annotated[str, typer.Argument(metavar="MODEL_ID", help="stored model id, as shown by `models list`")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="don't ask for confirmation")] = False,
) -> None:
    """Delete a model from the store."""
    deps.build_services().catalog.remove(model_id, confirmed=yes)
