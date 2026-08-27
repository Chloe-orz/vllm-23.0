# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.chat_completion.serving import (
    _extract_edge_cloud_media_items,
)
from vllm.inputs.engine import mm_input, tokens_input
from vllm.multimodal.inputs import MultiModalKwargsItems, PlaceholderRange

_HEX_DIGEST = "ab" * 32
_DIGEST = bytes([0xAB] * 32)

_OTHER_HEX_DIGEST = "cd" * 32
_OTHER_DIGEST = bytes([0xCD] * 32)


def _make_mm_input(
    num_tokens: int,
    mm_hashes: dict[str, list[str]],
    mm_placeholders: dict[str, list[PlaceholderRange]],
):
    return mm_input(
        prompt_token_ids=list(range(num_tokens)),
        mm_kwargs=MultiModalKwargsItems({}),
        mm_hashes=mm_hashes,
        mm_placeholders=mm_placeholders,
    )


def test_text_input_yields_no_media_items():
    engine_input = tokens_input(prompt_token_ids=[1, 2, 3])
    items = _extract_edge_cloud_media_items(
        engine_input, engine_input["prompt_token_ids"]
    )
    assert items == ()


def test_single_image_media_item():
    engine_input = _make_mm_input(
        10,
        {"image": [_HEX_DIGEST]},
        {"image": [PlaceholderRange(offset=2, length=4)]},
    )
    items = _extract_edge_cloud_media_items(
        engine_input, engine_input["prompt_token_ids"]
    )
    assert len(items) == 1
    item = items[0]
    assert item.modality == "image"
    assert item.digest == _DIGEST
    assert item.offset == 2
    assert item.length == 4


def test_items_sorted_by_modality_and_aligned_with_hashes():
    engine_input = _make_mm_input(
        20,
        {"image": [_HEX_DIGEST, _OTHER_HEX_DIGEST], "audio": [_OTHER_HEX_DIGEST]},
        {
            "image": [
                PlaceholderRange(offset=1, length=2),
                PlaceholderRange(offset=5, length=3),
            ],
            "audio": [PlaceholderRange(offset=10, length=4)],
        },
    )
    items = _extract_edge_cloud_media_items(
        engine_input, engine_input["prompt_token_ids"]
    )
    assert [(item.modality, item.offset, item.length) for item in items] == [
        ("audio", 10, 4),
        ("image", 1, 2),
        ("image", 5, 3),
    ]
    # Digests follow the per-modality item order of mm_hashes.
    assert [item.digest for item in items] == [_OTHER_DIGEST, _DIGEST, _OTHER_DIGEST]


def test_rejects_non_hex_digest():
    engine_input = _make_mm_input(
        10,
        {"image": ["not-a-hex-string"]},
        {"image": [PlaceholderRange(offset=0, length=4)]},
    )
    with pytest.raises(ValueError, match="not a hex string"):
        _extract_edge_cloud_media_items(engine_input, engine_input["prompt_token_ids"])


def test_accepts_sha512_digest():
    engine_input = _make_mm_input(
        10,
        {"image": ["ab" * 64]},
        {"image": [PlaceholderRange(offset=0, length=4)]},
    )
    items = _extract_edge_cloud_media_items(
        engine_input, engine_input["prompt_token_ids"]
    )
    assert len(items) == 1
    assert items[0].digest == bytes([0xAB] * 64)


def test_accepts_short_digest():
    engine_input = _make_mm_input(
        10,
        {"image": ["ab" * 16]},
        {"image": [PlaceholderRange(offset=0, length=4)]},
    )
    items = _extract_edge_cloud_media_items(
        engine_input, engine_input["prompt_token_ids"]
    )
    assert len(items) == 1
    assert items[0].digest == bytes([0xAB] * 16)


def test_rejects_empty_digest():
    engine_input = _make_mm_input(
        10,
        {"image": [""]},
        {"image": [PlaceholderRange(offset=0, length=4)]},
    )
    with pytest.raises(ValueError, match="empty digest"):
        _extract_edge_cloud_media_items(engine_input, engine_input["prompt_token_ids"])


def test_rejects_too_long_digest():
    engine_input = _make_mm_input(
        10,
        {"image": ["ab" * 65]},
        {"image": [PlaceholderRange(offset=0, length=4)]},
    )
    with pytest.raises(ValueError, match="expected at most 64 bytes"):
        _extract_edge_cloud_media_items(engine_input, engine_input["prompt_token_ids"])


@pytest.mark.parametrize(
    "placeholder",
    [
        PlaceholderRange(offset=-1, length=2),
        PlaceholderRange(offset=0, length=0),
        PlaceholderRange(offset=8, length=3),
    ],
)
def test_rejects_invalid_placeholder_range(placeholder: PlaceholderRange):
    engine_input = _make_mm_input(
        10,
        {"image": [_HEX_DIGEST]},
        {"image": [placeholder]},
    )
    with pytest.raises(ValueError, match="Invalid placeholder range"):
        _extract_edge_cloud_media_items(engine_input, engine_input["prompt_token_ids"])


def test_rejects_modality_mismatch():
    engine_input = _make_mm_input(
        10,
        {"image": [_HEX_DIGEST]},
        {"audio": [PlaceholderRange(offset=0, length=4)]},
    )
    with pytest.raises(ValueError, match="disagree on modalities"):
        _extract_edge_cloud_media_items(engine_input, engine_input["prompt_token_ids"])


def test_rejects_item_count_mismatch():
    engine_input = _make_mm_input(
        10,
        {"image": [_HEX_DIGEST, _OTHER_HEX_DIGEST]},
        {"image": [PlaceholderRange(offset=0, length=4)]},
    )
    with pytest.raises(ValueError, match="hashes but"):
        _extract_edge_cloud_media_items(engine_input, engine_input["prompt_token_ids"])
