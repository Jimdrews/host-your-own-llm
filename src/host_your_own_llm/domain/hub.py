"""What Hugging Face says about a model, and whether it fits the GPU."""

from collections.abc import Mapping
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

# Bytes per parameter for each safetensors dtype
DTYPE_BYTES = {
    "F64": 8, "I64": 8, "U64": 8,
    "F32": 4, "I32": 4, "U32": 4,
    "BF16": 2, "F16": 2, "I16": 2, "U16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1,
}


def weights_size(parameters: Mapping[str, int]) -> int:
    """Estimated weight size in bytes from safetensors parameter counts per dtype."""
    return sum(count * DTYPE_BYTES.get(dtype, 2) for dtype, count in parameters.items())


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


class ModelSummary(BaseModel):
    """A model from a Hub listing, with its weight size estimated from parameter counts."""

    model_config = ConfigDict(frozen=True)

    model_id: str
    downloads: int = 0
    likes: int = 0
    gated: bool = False
    size_bytes: int

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1024**3


class ModelSuggestion(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: ModelSummary
    fit: FitVerdict
