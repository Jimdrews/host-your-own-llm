"""Models offered by the setup questions."""

from pydantic import BaseModel, ConfigDict


class Preset(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    note: str


PRESETS = (
    Preset(model_id="Qwen/Qwen2.5-0.5B-Instruct", note="Tiny, good for testing"),
    Preset(model_id="Qwen/Qwen2.5-1.5B-Instruct", note="Small and fast"),
    Preset(model_id="Qwen/Qwen2.5-3B-Instruct", note="Balanced"),
    Preset(model_id="Qwen/Qwen2.5-7B-Instruct-AWQ", note="7B, 4-bit quantized"),
    Preset(model_id="Qwen/Qwen3-8B-AWQ", note="8B, 4-bit quantized, reasoning"),
)
