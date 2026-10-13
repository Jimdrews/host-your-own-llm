"""What Hugging Face says about a model, and whether it fits the GPU."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class ModelCheck(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    size_bytes: int
    gated: bool

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1024**3


class FitVerdict(StrEnum):
    UNKNOWN = "unknown"  # no GPU to compare against
    FITS = "fits"
    TIGHT = "tight"  # fits, with little room for the KV cache
    TOO_BIG = "too_big"
