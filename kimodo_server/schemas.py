import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LoadRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model: str
    encoder: str | None = None


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    model: str | None = None
    encoder: str | None = None
    block_swap: int | None = Field(default=None,ge=0,le=128)
    text_block_swap: int | None = Field(default=None,ge=0,le=32)
    prompts: list[str] = Field(min_length=1, max_length=16)
    durations: list[float] = Field(min_length=1, max_length=16)
    seed: int = Field(default=42, ge=0, le=2**31-1)
    seeds: list[int] | None = None
    steps: int = Field(default=100, ge=1, le=1000)
    transition_frames: int = Field(default=5, ge=1, le=60)
    cfg_text: float = Field(default=2.0, ge=0, le=10)
    cfg_constraints: float = Field(default=2.0, ge=0, le=10)
    post_processing: bool = False
    output_format: Literal['bvh', 'npz'] = 'bvh'
    standard_tpose: bool = True
    constraints: list[dict] = Field(default_factory=list, max_length=256)

    @model_validator(mode='after')
    def validate_segments(self):
        if len(self.prompts) != len(self.durations):
            raise ValueError('prompts and durations must have equal lengths')
        if any(not p.strip() or len(p) > 2000 for p in self.prompts):
            raise ValueError('each prompt must contain 1–2000 characters')
        if any(not math.isfinite(d) or d < 0.1 or d > 30 for d in self.durations):
            raise ValueError('each duration must be between 0.1 and 30 seconds')
        if sum(self.durations) > 120:
            raise ValueError('total duration must not exceed 120 seconds')
        if len(self.prompts) > 1 and any(round(d*30) < self.transition_frames for d in self.durations):
            raise ValueError('segments must be at least as long as transition_frames')
        if self.seeds is not None:
            if len(self.seeds) != len(self.prompts) or any(s < 0 or s > 2**31-1 for s in self.seeds):
                raise ValueError('seeds must contain one valid seed per prompt')
        return self
