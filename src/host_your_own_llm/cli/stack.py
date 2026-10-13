"""`serve`, `down` and `status`."""

from typing import Annotated

import typer

from ..ui import render
from . import deps
from .errors import reports_errors


@reports_errors
def serve(
    reconfigure: Annotated[bool, typer.Option("--reconfigure", help="ask the setup questions again")] = False,
) -> None:
    """Start the model store and vLLM (asks setup questions on first run)."""
    deps.build_services().stack.serve(reconfigure=reconfigure)


@reports_errors
def down(
    purge: Annotated[bool, typer.Option("--purge", help="also delete all stored models")] = False,
) -> None:
    """Stop everything; stored models are kept."""
    deps.build_services().stack.down(purge=purge)


@reports_errors
def status() -> None:
    """Show what's running."""
    render.print_status(deps.build_services().stack.status())
