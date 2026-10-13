"""The setup questions that choose a model and write .env."""

import secrets
from collections.abc import Sequence

from ..adapters.envfile import EnvFile
from ..domain import PRESETS, Preset, default_served_name
from ..ui import render
from ..ui.console import console
from ..ui.prompter import Prompter
from .vetting import ModelVetter


class SetupWizard:
    def __init__(self, env: EnvFile, vetter: ModelVetter, prompter: Prompter, presets: Sequence[Preset] = PRESETS):
        self.env = env
        self.vetter = vetter
        self.prompter = prompter
        self.presets = presets

    def configure(self) -> bool:
        """Ask the setup questions and write .env. Return False if the user backs out."""
        existing = self.env.load()

        console.rule("Model")
        self.vetter.gpu()
        model_id, token = self.choose_model(existing.hf_token)

        console.rule("Server")
        max_len = self.prompter.ask_int("Context length (tokens)", default=existing.max_model_len)
        served_name = self.prompter.ask("Model name for API requests", default=default_served_name(model_id))
        api_key = existing.vllm_api_key
        if self.prompter.confirm("Require an API key?", default=bool(api_key)):
            api_key = api_key or "sk-" + secrets.token_urlsafe(24)
            console.print(f"API key: [bold]{api_key}[/bold]")
        else:
            api_key = ""

        config = existing.model_copy(update={
            "s3_secret_key": existing.s3_secret_key or secrets.token_urlsafe(24),
            "model": existing.uri_for(model_id),
            "hf_token": token,
            "max_model_len": max_len,
            "served_model_name": served_name,
            "vllm_api_key": api_key,
        })

        console.rule("Summary")
        console.print(render.summary_table(config))
        if not self.prompter.confirm("Write this to .env?", default=True):
            return False
        self.env.save(config)
        console.print("[green]Saved .env[/green]")
        return True

    def choose_model(self, token: str) -> tuple[str, str]:
        """Show the presets and return (model id, token)."""
        console.print(render.presets_table(self.presets))
        other = len(self.presets) + 1
        while True:
            choice = self.prompter.ask_int("Choose a model", default=1)
            if 1 <= choice <= len(self.presets):
                model_id = self.presets[choice - 1].model_id
            elif choice == other:
                model_id = self.prompter.ask("Hugging Face model id (e.g. org/name)")
            else:
                console.print("[red]Pick one of the numbers above.[/red]")
                continue
            checked_token = self.vetter.vet(model_id, token)
            if checked_token is not None:
                return model_id, checked_token
