# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import uuid

import torch

MASK_64_BITS = (1 << 64) - 1


def is_prefill_only_lwd(vllm_config) -> bool:
    """Whether the prefill_only LWD data plane is enabled.

    Reads ``additional_config["lwd_config"]`` (a plain dict, no vllm
    config-schema change).  Init-order independent: only inspects the config
    object, never imports vllm_ascend (no import cycle).
    """
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    lwd_config = additional_config.get("lwd_config") or {}
    return bool(lwd_config.get("enabled", False)) and \
        lwd_config.get("mode", None) == "prefill_only"


def random_uuid() -> str:
    return f"{uuid.uuid4().int & MASK_64_BITS:016x}"  # 16 hex chars


def length_from_prompt_token_ids_or_embeds(
    prompt_token_ids: list[int] | torch.Tensor | None,
    prompt_embeds: torch.Tensor | None,
) -> int:
    """Calculate the request length (in number of tokens) give either
    prompt_token_ids or prompt_embeds.
    """
    prompt_token_len = None if prompt_token_ids is None else len(prompt_token_ids)
    prompt_embeds_len = None if prompt_embeds is None else len(prompt_embeds)

    if prompt_token_len is None:
        if prompt_embeds_len is None:
            raise ValueError("Neither prompt_token_ids nor prompt_embeds were defined.")
        return prompt_embeds_len
    else:
        if prompt_embeds_len is not None and prompt_embeds_len != prompt_token_len:
            raise ValueError(
                "Prompt token ids and prompt embeds had different lengths"
                f" prompt_token_ids={prompt_token_len}"
                f" prompt_embeds={prompt_embeds_len}"
            )
        return prompt_token_len


def is_moe_layer(module: torch.nn.Module) -> bool:
    # TODO(bnell): Should use isinstance but can't due to circular dependencies.
    def _check_bases(cls):
        if cls.__name__ == "FusedMoE":
            return True

        for b in cls.__bases__:
            if _check_bases(b):
                return True

    return _check_bases(module.__class__)
