"""`llm`: run the local stack (serve, down, status, models)."""

import typer

from . import models, stack

app = typer.Typer(
    name="llm",
    help="Run your own LLM locally with vLLM.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
app.command()(stack.serve)
app.command()(stack.down)
app.command()(stack.status)
app.add_typer(models.app, name="models")


def main() -> None:
    app(prog_name="llm")
