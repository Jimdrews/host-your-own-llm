"""The stack's settings, as stored in .env."""

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator
from pydantic_core import PydanticUseDefault


def default_served_name(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


class StackConfig(BaseModel):
    """Settings read from .env. Field aliases are the .env keys."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    s3_access_key: str = Field("modelstore", alias="S3_ACCESS_KEY")
    s3_secret_key: str = Field("", alias="S3_SECRET_KEY")
    s3_bucket: str = Field("models", alias="S3_BUCKET")
    model: str = Field("", alias="MODEL")
    hf_token: str = Field("", alias="HF_TOKEN")
    max_model_len: int = Field(8192, alias="MAX_MODEL_LEN")
    gpu_memory_utilization: float = Field(0.90, alias="GPU_MEMORY_UTILIZATION")
    served_model_name: str = Field("", alias="SERVED_MODEL_NAME")
    vllm_api_key: str = Field("", alias="VLLM_API_KEY")
    extra_args: str = Field("", alias="EXTRA_ARGS")
    # Read for the API URL, but left to the user: setup never writes it
    api_port: int | None = Field(None, alias="API_PORT", exclude=True)

    @field_validator("s3_access_key", "s3_bucket", "max_model_len", "gpu_memory_utilization", "api_port", mode="before")
    @classmethod
    def _blank_means_default(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            raise PydanticUseDefault()
        return value

    @field_serializer("gpu_memory_utilization")
    def _two_decimals(self, value: float) -> str:
        return f"{value:.2f}"

    @property
    def api_url(self) -> str:
        return f"http://localhost:{self.api_port or 8000}"

    @property
    def active_model_id(self) -> str:
        return self.model.removeprefix(f"s3://{self.s3_bucket}/")

    @property
    def display_name(self) -> str:
        """The name API requests use: vLLM falls back to MODEL when no served name is set."""
        return self.served_model_name or self.model

    def uri_for(self, model_id: str) -> str:
        return f"s3://{self.s3_bucket}/{model_id}"

    def to_env(self) -> dict[str, str]:
        """The .env lines setup owns; other keys in the file are left alone."""
        return {key: str(value) for key, value in self.model_dump(by_alias=True).items()}
