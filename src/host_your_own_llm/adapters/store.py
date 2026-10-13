"""The model store: an S3 bucket managed by the `store` tool container (image/model_store.py)."""

import uuid

from .compose import Compose


class ModelStore:
    def __init__(self, compose: Compose):
        self.compose = compose

    def start(self) -> None:
        self.compose.up("s3", wait=True)

    def list_models(self) -> list[str]:
        return self._run("list", quiet=True).split()

    def seed(self, model_id: str | None = None) -> None:
        """Copy a model from Hugging Face; with no id, the one MODEL names."""
        self._run("seed", *([model_id] if model_id else []))

    def rm(self, model_id: str) -> None:
        self._run("rm", model_id)

    def _run(self, *args: str, quiet: bool = False) -> str:
        name = f"host-your-own-llm-store-{uuid.uuid4().hex[:8]}"
        try:
            return self.compose.run("run", "--rm", "-T", "--name", name, "store", *args, quiet=quiet)
        except KeyboardInterrupt:
            # Ctrl+C stops `docker compose run` but can leave its container copying
            self.compose.remove_container(name)
            raise
