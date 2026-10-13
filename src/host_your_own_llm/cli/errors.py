from collections.abc import Callable
from functools import wraps

import typer

from ..errors import CliError
from ..ui.console import console


def reports_errors[**P](command: Callable[P, None]) -> Callable[P, None]:
    """Show CliError and Ctrl+C as a short message and exit 1, instead of a traceback."""

    @wraps(command)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
        try:
            command(*args, **kwargs)
        except CliError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from None
        except (KeyboardInterrupt, EOFError):
            console.print("\nCancelled.")
            raise typer.Exit(1) from None

    return wrapper
