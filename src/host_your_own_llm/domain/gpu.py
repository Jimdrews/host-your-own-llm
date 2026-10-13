from pydantic import BaseModel, ConfigDict


class GpuInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    vram_gb: float
