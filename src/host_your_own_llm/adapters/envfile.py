"""Read and update .env without disturbing comments or keys we don't own."""

from pathlib import Path

from ..domain import StackConfig
from ..errors import CliError


def read_env(path: Path) -> dict[str, str]:
    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
    return env


def update_env(path: Path, **changes: str) -> None:
    """Set the given keys in .env, leaving every other line as it is."""
    lines = path.read_text().splitlines() if path.exists() else []
    pending = dict(changes)
    for i, line in enumerate(lines):
        key = line.split("=", 1)[0].strip()
        if key in pending:
            lines[i] = f"{key}={pending.pop(key)}"
    lines += [f"{k}={v}" for k, v in pending.items()]
    path.write_text("\n".join(lines) + "\n")


class EnvFile:
    """The project's .env, loaded as a StackConfig."""

    def __init__(self, path: Path):
        self.path = path

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> StackConfig:
        return StackConfig.model_validate(read_env(self.path) if self.exists() else {})

    def require(self) -> StackConfig:
        if not self.exists():
            raise CliError("Not configured yet. Run `uv run llm serve` first.")
        return self.load()

    def update(self, **changes: str) -> None:
        """Set raw .env keys, e.g. update(MODEL=...)."""
        update_env(self.path, **changes)

    def save(self, config: StackConfig) -> None:
        update_env(self.path, **config.to_env())
