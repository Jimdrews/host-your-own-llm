"""Wire the real adapters into the services. Tests replace build_services to use fakes."""

from dataclasses import dataclass
from pathlib import Path

from ..adapters.compose import Compose
from ..adapters.envfile import EnvFile
from ..adapters.huggingface import HubClient
from ..adapters.store import ModelStore
from ..project import find_project_root
from ..services.catalog import ModelCatalog
from ..services.discovery import ModelDiscovery
from ..services.setup import SetupWizard
from ..services.stack import StackService
from ..services.vetting import ModelVetter
from ..ui.prompter import RichPrompter


@dataclass
class Services:
    stack: StackService
    catalog: ModelCatalog
    discovery: ModelDiscovery


def build_services(root: Path | None = None) -> Services:
    root = root or find_project_root()
    env = EnvFile(root / ".env")
    compose = Compose(root)
    store = ModelStore(compose)
    prompter = RichPrompter()
    hub = HubClient()
    vetter = ModelVetter(hub, prompter)
    discovery = ModelDiscovery(hub, vetter)
    wizard = SetupWizard(env, vetter, prompter, discovery)
    stack = StackService(compose, store, env, prompter, configure=wizard.configure)
    catalog = ModelCatalog(env, store, vetter, prompter, stack)
    return Services(stack=stack, catalog=catalog, discovery=discovery)
