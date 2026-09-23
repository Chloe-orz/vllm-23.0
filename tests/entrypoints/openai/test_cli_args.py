# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.models.protocol import LoRAModulePath
from vllm.utils.argparse_utils import FlexibleArgumentParser

from ...utils import VLLM_PATH

LORA_MODULE = {
    "name": "module2",
    "path": "/path/to/module2",
    "base_model_name": "llama",
}
CHATML_JINJA_PATH = VLLM_PATH / "examples/template_chatml.jinja"
assert CHATML_JINJA_PATH.exists()


def _build_vllm_parsers():
    vllm_parser = FlexibleArgumentParser()
    subparsers = vllm_parser.add_subparsers()
    serve_parser = subparsers.add_parser("serve")
    make_arg_parser(serve_parser)
    return {"vllm": vllm_parser, "vllm serve": serve_parser}


@pytest.fixture
def vllm_parser():
    return _build_vllm_parsers()["vllm"]


@pytest.fixture
def serve_parser():
    return _build_vllm_parsers()["vllm serve"]


### Test config parsing
def test_config_arg_parsing(serve_parser, cli_config_file):
    args = serve_parser.parse_args([])
    assert args.port == 8000
    args = serve_parser.parse_args(["--config", cli_config_file])
    assert args.port == 12312
    args = serve_parser.parse_args(
        [
            "--config",
            cli_config_file,
            "--port",
            "9000",
        ]
    )
    assert args.port == 9000
    args = serve_parser.parse_args(
        [
            "--port",
            "9000",
            "--config",
            cli_config_file,
        ]
    )
    assert args.port == 9000


### Tests for LoRA module parsing
def test_valid_key_value_format(serve_parser):
    # Test old format: name=path
    args = serve_parser.parse_args(
        [
            "--lora-modules",
            "module1=/path/to/module1",
        ]
    )
    expected = [LoRAModulePath(name="module1", path="/path/to/module1")]
    assert args.lora_modules == expected


def test_valid_json_format(serve_parser):
    # Test valid JSON format input
    args = serve_parser.parse_args(
        [
            "--lora-modules",
            json.dumps(LORA_MODULE),
        ]
    )
    expected = [
        LoRAModulePath(name="module2", path="/path/to/module2", base_model_name="llama")
    ]
    assert args.lora_modules == expected


def test_invalid_json_format(serve_parser):
    # Test invalid JSON format input, missing closing brace
    with pytest.raises(SystemExit):
        serve_parser.parse_args(
            ["--lora-modules", '{"name": "module3", "path": "/path/to/module3"']
        )


def test_invalid_type_error(serve_parser):
    # Test type error when values are not JSON or key=value
    with pytest.raises(SystemExit):
        serve_parser.parse_args(
            [
                "--lora-modules",
                "invalid_format",  # This is not JSON or key=value format
            ]
        )


def test_invalid_json_field(serve_parser):
    # Test valid JSON format but missing required fields
    with pytest.raises(SystemExit):
        serve_parser.parse_args(
            [
                "--lora-modules",
                '{"name": "module4"}',  # Missing required 'path' field
            ]
        )


def test_empty_values(serve_parser):
    # Test when no LoRA modules are provided
    args = serve_parser.parse_args(["--lora-modules", ""])
    assert args.lora_modules == []


def test_multiple_valid_inputs(serve_parser):
    # Test multiple valid inputs (both old and JSON format)
    args = serve_parser.parse_args(
        [
            "--lora-modules",
            "module1=/path/to/module1",
            json.dumps(LORA_MODULE),
        ]
    )
    expected = [
        LoRAModulePath(name="module1", path="/path/to/module1"),
        LoRAModulePath(
            name="module2", path="/path/to/module2", base_model_name="llama"
        ),
    ]
    assert args.lora_modules == expected


### Tests for serve argument validation that run prior to loading
def test_enable_auto_choice_passes_without_tool_call_parser(serve_parser):
    """Ensure validation fails if tool choice is enabled with no call parser"""
    # If we enable-auto-tool-choice, explode with no tool-call-parser
    args = serve_parser.parse_args(args=["--enable-auto-tool-choice"])
    with pytest.raises(TypeError):
        validate_parsed_serve_args(args)


def test_enable_auto_choice_passes_with_tool_call_parser(serve_parser):
    """Ensure validation passes with tool choice enabled with a call parser"""
    args = serve_parser.parse_args(
        args=[
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "mistral",
        ]
    )
    validate_parsed_serve_args(args)


def test_enable_auto_choice_fails_with_enable_reasoning(serve_parser):
    """Ensure validation fails if reasoning is enabled with auto tool choice"""
    args = serve_parser.parse_args(
        args=[
            "--enable-auto-tool-choice",
            "--reasoning-parser",
            "deepseek_r1",
        ]
    )
    with pytest.raises(TypeError):
        validate_parsed_serve_args(args)


def test_passes_with_reasoning_parser(serve_parser):
    """Ensure validation passes if reasoning is enabled
    with a reasoning parser"""
    args = serve_parser.parse_args(
        args=[
            "--reasoning-parser",
            "deepseek_r1",
        ]
    )
    validate_parsed_serve_args(args)


def test_chat_template_validation_for_happy_paths(serve_parser):
    """Ensure validation passes if the chat template exists"""
    args = serve_parser.parse_args(
        args=["--chat-template", CHATML_JINJA_PATH.absolute().as_posix()]
    )
    validate_parsed_serve_args(args)


def test_chat_template_validation_for_sad_paths(serve_parser):
    """Ensure validation fails if the chat template doesn't exist"""
    args = serve_parser.parse_args(args=["--chat-template", "does/not/exist"])
    with pytest.raises(ValueError):
        validate_parsed_serve_args(args)


@pytest.mark.parametrize(
    "cli_args, expected_middleware",
    [
        (
            ["--middleware", "middleware1", "--middleware", "middleware2"],
            ["middleware1", "middleware2"],
        ),
        ([], []),
    ],
)
def test_middleware(serve_parser, cli_args, expected_middleware):
    """Ensure multiple middleware args are parsed properly"""
    args = serve_parser.parse_args(args=cli_args)
    assert args.middleware == expected_middleware


def test_default_chat_template_kwargs_parsing(serve_parser):
    """Ensure default_chat_template_kwargs JSON is parsed correctly"""
    args = serve_parser.parse_args(
        args=["--default-chat-template-kwargs", '{"enable_thinking": false}']
    )
    assert args.default_chat_template_kwargs == {"enable_thinking": False}


def test_default_chat_template_kwargs_complex(serve_parser):
    """Ensure complex default_chat_template_kwargs JSON is parsed correctly"""
    kwargs_json = '{"enable_thinking": false, "custom_param": "value", "num": 42}'
    args = serve_parser.parse_args(args=["--default-chat-template-kwargs", kwargs_json])
    assert args.default_chat_template_kwargs == {
        "enable_thinking": False,
        "custom_param": "value",
        "num": 42,
    }


def test_default_chat_template_kwargs_default_none(serve_parser):
    """Ensure default_chat_template_kwargs defaults to None"""
    args = serve_parser.parse_args(args=[])
    assert args.default_chat_template_kwargs is None


def test_default_chat_template_kwargs_invalid_json(serve_parser):
    """Ensure invalid JSON raises an error"""
    with pytest.raises(SystemExit):
        serve_parser.parse_args(
            args=["--default-chat-template-kwargs", "not valid json"]
        )


@pytest.mark.parametrize(
    "args, raises",
    [
        (["user/model"], None),
        (["user/model", "--served-model-name", "model"], None),
        (["--served-model-name", "model", "user/model"], ValueError),
        (["--served-model-name", "model", "--config", "config.yaml"], None),
        (["--served-model-name", "model", "--config", "config.yaml"], ValueError),
    ],
    ids=[
        "model_tag_only",
        "model_tag_with_served_model_name",
        "served_model_name_before_model_tag",
        "served_model_name_with_model_in_config",
        "served_model_name_with_no_model_in_config",
    ],
)
def test_served_model_name_parsing(tmp_path, vllm_parser, args, raises):
    """Ensure that users don't misuse --served-model-name and end up with the default
    model tag instead of the one they intended to serve."""
    # Call the serve subparser
    args.insert(0, "serve")
    # Create a dummy config file if the test case includes it
    if "config.yaml" in args:
        # Create a dummy config file if the test case includes it
        config_path = tmp_path / "config.yaml"
        config_path.write_text("model: user/model" if raises is None else "port: 8000")
        args[args.index("config.yaml")] = config_path.as_posix()
    # Do the parsing and check for expected exceptions or values
    if raises is None:
        parsed_args = vllm_parser.parse_args(args=args)
        expected = "user/model"
        assert parsed_args.model_tag == expected or parsed_args.model == expected
    else:
        with pytest.raises(raises):
            vllm_parser.parse_args(args=args)


### Tests for LoRA target modules parsing
def test_lora_target_modules_single(serve_parser):
    """Test parsing single lora-target-modules argument"""
    args = serve_parser.parse_args(
        args=["--enable-lora", "--lora-target-modules", "o_proj"]
    )
    assert args.lora_target_modules == ["o_proj"]


def test_lora_target_modules_multiple(serve_parser):
    """Test parsing multiple lora-target-modules arguments"""
    args = serve_parser.parse_args(
        args=[
            "--enable-lora",
            "--lora-target-modules",
            "o_proj",
            "qkv_proj",
            "down_proj",
        ]
    )
    assert args.lora_target_modules == ["o_proj", "qkv_proj", "down_proj"]


def test_lora_target_modules_default_none(serve_parser):
    """Test that lora-target-modules defaults to None"""
    args = serve_parser.parse_args(args=[])
    assert args.lora_target_modules is None


@pytest.mark.parametrize("api_count", [1, 2])
@pytest.mark.parametrize("instance_id", [0, 1])
def test_lwd_api_options_reach_engine_args_without_yaml_io(
    serve_parser, monkeypatch, api_count, instance_id
):
    from pathlib import Path

    from vllm.engine.arg_utils import AsyncEngineArgs

    flag = "--api-server-rpc-port" if instance_id == 0 else "--api-server-attach"
    additional = {
        "enable_cpu_binding": True,
        "lwd_config": {
            "path": "/does/not/exist/lwd_config.yaml",
            "role": "edge",
            "instance_id": instance_id,
        },
    }
    args = serve_parser.parse_args(
        [
            "--api-server-count",
            str(api_count),
            "--additional-config",
            json.dumps(additional),
            flag,
            "10.0.0.1:29550",
        ]
    )
    validate_parsed_serve_args(args)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", lambda *a, **kw: pytest.fail("early YAML read"))
        engine_args = AsyncEngineArgs.from_cli_args(args)
    api = engine_args.lwd_api_config
    assert api is not None
    assert getattr(api, flag[2:].replace("-", "_")) == "10.0.0.1:29550"
    assert engine_args.additional_config == additional
    assert args.api_server_count == api_count
    assert args.data_parallel_size == 1


@pytest.mark.parametrize(
    "extra",
    [
        ["--api-server-rpc-port", "bad"],
        ["--api-server-attach", "10.0.0.1:29550"],  # primary cannot attach
        ["--api-server-rpc-port", "10.0.0.1:29550", "--headless"],
        ["--api-server-rpc-port", "10.0.0.1:29550", "--grpc"],
        [
            "--api-server-rpc-port",
            "10.0.0.1:29550",
            "--api-server-attach",
            "10.0.0.1:29550",
        ],
    ],
)
def test_lwd_api_invalid_cli_options(serve_parser, extra):
    args = serve_parser.parse_args(
        [
            "--additional-config",
            json.dumps(
                {"lwd_config": {
                    "path": "/unused.yaml",
                    "role": "edge",
                    "instance_id": 0,
                }}
            ),
            *extra,
        ]
    )
    with pytest.raises(ValueError, match="LWD"):
        validate_parsed_serve_args(args)


def test_lwd_api_options_without_lwd(serve_parser):
    args = serve_parser.parse_args(["--api-server-rpc-port", "10.0.0.1:29550"])
    with pytest.raises(ValueError, match="require lwd_config.path"):
        validate_parsed_serve_args(args)


def test_lwd_api_absent_preserves_normal_frontend(serve_parser):
    from vllm.engine.arg_utils import AsyncEngineArgs

    args = serve_parser.parse_args([])
    validate_parsed_serve_args(args)
    engine_args = AsyncEngineArgs.from_cli_args(args)
    assert engine_args.lwd_api_config is not None
    assert not engine_args.lwd_api_config.requested
    assert args.api_server_count is None
