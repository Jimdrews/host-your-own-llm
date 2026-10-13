"""The models in the store: list, add, switch and remove."""

from ..adapters.envfile import EnvFile
from ..adapters.store import ModelStore
from ..domain import default_served_name
from ..errors import CliError
from ..ui.console import console
from ..ui.prompter import Prompter
from .stack import StackService
from .vetting import ModelVetter


class ModelCatalog:
    def __init__(self, env: EnvFile, store: ModelStore, vetter: ModelVetter, prompter: Prompter, stack: StackService):
        self.env = env
        self.store = store
        self.vetter = vetter
        self.prompter = prompter
        self.stack = stack

    def active_model_id(self) -> str:
        return self.env.require().active_model_id

    def list_models(self) -> list[str]:
        self.env.require()
        self.store.start()
        return self.store.list_models()

    def ensure_stored(self, model_id: str) -> None:
        """Make sure a model is in the store, copying it from Hugging Face if needed."""
        config = self.env.require()
        self.store.start()
        if model_id in self.store.list_models():
            console.print(f"{model_id} is already in the model store.")
            return
        token = self.vetter.vet(model_id, config.hf_token)
        if token is None:
            raise CliError(f"Not adding {model_id}.")
        if token != config.hf_token:
            self.env.update(HF_TOKEN=token)
        self.store.seed(model_id)

    def use(self, model_id: str, served_name: str | None = None) -> None:
        """Make a model the one vLLM serves, restarting vLLM if it is running."""
        self.ensure_stored(model_id)
        config = self.env.require()
        self.env.update(MODEL=config.uri_for(model_id), SERVED_MODEL_NAME=served_name or default_served_name(model_id))
        if self.stack.vllm_running():
            self.stack.start_vllm(self.env.load())
        else:
            console.print(f"Now using {model_id}. Start it with `uv run llm serve`.")

    def remove(self, model_id: str, confirmed: bool = False) -> None:
        if model_id == self.env.require().active_model_id:
            raise CliError(f"{model_id} is the active model; switch with `uv run llm models use` first.")
        if not confirmed and not self.prompter.confirm(f"Delete {model_id} from the model store?", default=False):
            raise CliError("Nothing deleted.")
        self.store.start()
        self.store.rm(model_id)
