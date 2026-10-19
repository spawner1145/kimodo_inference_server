import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelProfile(BaseModel):
    model_config = ConfigDict(extra='forbid')
    label: str
    path: str
    precision: Literal['fp32', 'fp16', 'bf16', 'int8_convrot'] = 'fp32'
    skeleton: Literal['SOMA','SMPLX','G1'] = 'SOMA'
    repo_id: str | None = None
    download_source: Literal['huggingface', 'modelscope'] = 'huggingface'


class EncoderConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    label: str = 'Text encoder'
    kind: Literal['awq_llm2vec', 'nf4_llm2vec', 'llm2vec'] = 'awq_llm2vec'
    base: str
    mntp: str | None = None
    supervised: str | None = None
    block_swap: int = Field(default=0, ge=0, le=32)
    repo_id: str | None = None
    mntp_repo_id: str | None = None
    supervised_repo_id: str | None = None
    download_source: Literal['huggingface', 'modelscope'] = 'huggingface'

    @model_validator(mode='after')
    def check_adapters(self):
        if self.kind!='nf4_llm2vec' and (not self.mntp or not self.supervised):
            raise ValueError('AWQ and full-precision LLM2Vec require both adapter paths')
        return self


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    host: str = '127.0.0.1'
    port: int = Field(default=6600, ge=1, le=65535)
    device: str = 'cuda'
    models_dir: Path = Path('models')
    data_dir: Path = Path('data')
    default_model: str = 'soma-fp32'
    offload: bool = True
    tf32: bool = False
    cpu_threads: int = Field(default=4, ge=1, le=64)
    queue_size: int = Field(default=8, ge=1, le=32)
    retention_hours: int = Field(default=168, ge=1, le=8760)
    default_encoder: str = 'awq'
    encoders: dict[str,EncoderConfig]
    block_swap: int = Field(default=0,ge=0,le=128)
    models: dict[str, ModelProfile]
    inference: 'InferenceDefaults' = Field(default_factory=lambda: InferenceDefaults())

    @model_validator(mode='after')
    def check_default(self):
        if self.default_model not in self.models:
            raise ValueError('default_model must exist in models')
        if self.default_encoder not in self.encoders:
            raise ValueError('default_encoder must exist in encoders')
        return self


class InferenceDefaults(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, le=2**31-1)
    steps: int = Field(default=100, ge=1, le=1000)
    transition_frames: int = Field(default=5, ge=1, le=60)
    cfg_text: float = Field(default=2, ge=0, le=10)
    cfg_constraints: float = Field(default=2, ge=0, le=10)
    post_processing: bool = False
    standard_tpose: bool = True
    output_format: Literal['bvh', 'npz'] = 'bvh'


ServerConfig.model_rebuild()


def resolve_config(document, path):
    path = Path(path).resolve()
    config = ServerConfig.model_validate(document)
    for name in ('models_dir', 'data_dir'):
        value = getattr(config, name).expanduser()
        setattr(config, name, (path.parent / value).resolve())
    return config


def load_config(path):
    path = Path(path)
    # Source checkouts keep machine-specific settings out of Git. Create only
    # the default config, never overwrite one or silently replace custom paths.
    example = path.with_name('config.example.json')
    if not path.exists() and path.name == 'config.json' and example.is_file():
        document = example.read_text(encoding='utf-8-sig')
        resolve_config(json.loads(document), path)
        try:
            with path.open('x', encoding='utf-8') as stream:
                stream.write(document)
        except FileExistsError:
            pass
    return resolve_config(json.loads(path.read_text(encoding='utf-8-sig')), path)


def api_token():
    return os.environ.get('KIMODO_API_TOKEN', '').strip()
