"""Questions for the user, behind an interface so services can be tested with scripted answers."""

import sys
from typing import Protocol

from rich.prompt import Confirm, IntPrompt, Prompt

from .console import console


class Prompter(Protocol):
    def confirm(self, question: str, default: bool) -> bool: ...

    def ask(self, question: str, default: str | None = None) -> str: ...

    def ask_int(self, question: str, default: int) -> int: ...

    def ask_secret(self, question: str) -> str | None:
        """Ask without echoing; None when there is no terminal to ask on."""
        ...


class RichPrompter:
    def confirm(self, question: str, default: bool) -> bool:
        return Confirm.ask(question, default=default, console=console)

    def ask(self, question: str, default: str | None = None) -> str:
        if default is None:
            return Prompt.ask(question, console=console).strip()
        return Prompt.ask(question, default=default, console=console).strip()

    def ask_int(self, question: str, default: int) -> int:
        return IntPrompt.ask(question, default=default, console=console)

    def ask_secret(self, question: str) -> str | None:
        if not sys.stdin.isatty():
            # The hidden prompt reads the terminal directly and would wait forever in a script
            return None
        return Prompt.ask(question, password=True, console=console).strip()
