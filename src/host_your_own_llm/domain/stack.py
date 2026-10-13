"""The state of the compose services."""

from pydantic import BaseModel, ConfigDict, Field

from .config import StackConfig

SERVICES = ("s3", "vllm")


class ServiceState(BaseModel):
    """One line of `docker compose ps --format json`."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    service: str = Field(alias="Service")
    state: str = Field(alias="State")
    health: str = Field("", alias="Health")

    @property
    def effective(self) -> str:
        return self.health or self.state


class StackStatus(BaseModel):
    services: dict[str, str] = {}
    config: StackConfig | None = None
    api_up: bool = False

    def state(self, service: str) -> str:
        return self.services.get(service, "stopped")
