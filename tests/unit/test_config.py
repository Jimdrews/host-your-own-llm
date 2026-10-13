import pytest
from pydantic import ValidationError

from host_your_own_llm.domain import StackConfig, default_served_name


def test_blank_values_take_defaults():
    config = StackConfig.model_validate({"API_PORT": "", "MAX_MODEL_LEN": "", "S3_BUCKET": " ", "GPU_MEMORY_UTILIZATION": ""})
    assert config.api_port is None
    assert config.max_model_len == 8192
    assert config.s3_bucket == "models"
    assert config.gpu_memory_utilization == 0.90


def test_unknown_keys_are_ignored():
    config = StackConfig.model_validate({"COMPOSE_PROJECT_NAME": "x", "MODEL": "org/m"})
    assert config.model == "org/m"


def test_bad_numbers_are_rejected():
    with pytest.raises(ValidationError):
        StackConfig.model_validate({"MAX_MODEL_LEN": "lots"})


def test_api_url():
    assert StackConfig().api_url == "http://localhost:8000"
    assert StackConfig.model_validate({"API_PORT": "18000"}).api_url == "http://localhost:18000"


def test_active_model_id():
    config = StackConfig.model_validate({"MODEL": "s3://bucket/org/model", "S3_BUCKET": "bucket"})
    assert config.active_model_id == "org/model"
    assert config.uri_for("a/b") == "s3://bucket/a/b"


def test_display_name_falls_back_to_model():
    assert StackConfig(MODEL="org/m").display_name == "org/m"
    assert StackConfig(MODEL="org/m", SERVED_MODEL_NAME="m").display_name == "m"


def test_to_env_writes_setup_keys_only():
    env = StackConfig.model_validate({"API_PORT": "18000", "GPU_MEMORY_UTILIZATION": "0.5"}).to_env()
    assert list(env) == [
        "S3_ACCESS_KEY", "S3_SECRET_KEY", "S3_BUCKET", "MODEL", "HF_TOKEN", "MAX_MODEL_LEN",
        "GPU_MEMORY_UTILIZATION", "SERVED_MODEL_NAME", "VLLM_API_KEY", "EXTRA_ARGS",
    ]
    assert env["GPU_MEMORY_UTILIZATION"] == "0.50"
    assert env["MAX_MODEL_LEN"] == "8192"


def test_round_trip():
    config = StackConfig(MODEL="s3://models/org/m", HF_TOKEN="hf_1", MAX_MODEL_LEN=4096)
    assert StackConfig.model_validate(config.to_env()) == config


def test_default_served_name():
    assert default_served_name("Qwen/Qwen2.5-0.5B-Instruct") == "qwen2.5-0.5b-instruct"
