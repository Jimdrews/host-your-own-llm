"""Start, stop and inspect the compose stack."""

import time
from collections.abc import Callable

from ..adapters.compose import check_docker
from ..adapters.envfile import EnvFile
from ..adapters.vllm_api import is_healthy
from ..domain import StackConfig, StackStatus
from ..errors import CliError
from ..ports import ComposePort, StorePort
from ..ui import render
from ..ui.console import console
from ..ui.prompter import Prompter

HEALTH_TIMEOUT = 900
POLL_INTERVAL = 3


class StackService:
    def __init__(
        self,
        compose: ComposePort,
        store: StorePort,
        env: EnvFile,
        prompter: Prompter,
        configure: Callable[[], bool],
        *,
        docker_check: Callable[[], bool] = check_docker,
        health: Callable[[str], bool] = is_healthy,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        timeout: float = HEALTH_TIMEOUT,
    ):
        self.compose = compose
        self.store = store
        self.env = env
        self.prompter = prompter
        self.configure = configure
        self.docker_check = docker_check
        self.health = health
        self.clock = clock
        self.sleep = sleep
        self.timeout = timeout

    def serve(self, reconfigure: bool = False) -> None:
        if not self.docker_check():
            console.print("[yellow]Docker doesn't list the NVIDIA runtime; vLLM needs GPU access.[/yellow]")
        needs_setup = reconfigure or not self.env.exists()
        if needs_setup and not self.configure():
            raise CliError("Setup cancelled.")
        config = self.env.load()

        with console.status("Building the image (the first build downloads ~10 GB)..."):
            self.compose.build("vllm")
        with console.status("Starting the model store..."):
            self.store.start()
        self.store.seed()
        self.start_vllm(config)

    def vllm_running(self) -> bool:
        return self.compose.is_running("vllm")

    def start_vllm(self, config: StackConfig) -> None:
        """Start (or restart with new settings) vLLM and wait until it answers."""
        self.compose.up("vllm")
        start = self.clock()
        with console.status("Loading the model...") as status:
            while not self.health(config.api_url):
                elapsed = self.clock() - start
                if elapsed > self.timeout:
                    self.compose.logs("vllm", tail=40)
                    raise CliError("vLLM didn't become ready in time; see the logs above.")
                status.update(f"Loading the model... {elapsed:.0f}s")
                self.sleep(POLL_INTERVAL)
        render.print_usage(config)

    def down(self, purge: bool = False) -> None:
        if purge:
            if not self.prompter.confirm("Delete every model in the model store?", default=False):
                raise CliError("Nothing deleted.")
            self.compose.down(volumes=True)
            console.print("Stopped, and deleted the stored models.")
        else:
            self.compose.down()
            console.print("Stopped. Models are kept; `uv run llm serve` starts again.")

    def status(self) -> StackStatus:
        services = {state.service: state.effective for state in self.compose.ps()}
        config = self.env.load() if self.env.exists() else None
        api_up = "vllm" in services and self.health((config or StackConfig()).api_url)
        if api_up:
            # Docker's healthcheck only runs every 30s, so ask the API directly
            services["vllm"] = "healthy"
        return StackStatus(services=services, config=config, api_up=api_up)
