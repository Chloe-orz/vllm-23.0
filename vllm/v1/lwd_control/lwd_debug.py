# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LWD 临时调试日志(独立模块,便于整体下线)。

接入方式:功能代码里只保留单行 ``LwdDebug.xxx(...)`` 调用(带
``# [lwd-debug]`` 注释标记);下线时删除本文件并移除带标记的调用行
即可,功能逻辑零残留。``LwdDebug.ENABLED = False`` 可整体静默。

日志标签:
  [lwd-finish-dbg]  云侧:请求准入停止参数;每步 token 进度与 finish
  [lwd-token-dbg]   边侧:每拍实际交付的 token(解码为文字)
"""

from __future__ import annotations


class LwdDebug:
    """无状态静态调试器;tokenizer 懒加载后缓存于类属性。"""

    ENABLED = True
    _tokenizer = None
    _layers_reported = False

    @classmethod
    def _log(cls, msg: str, *args) -> None:
        if not cls.ENABLED:
            return
        from vllm.logger import init_logger

        init_logger(__name__).info(msg, *args)

    @staticmethod
    def _lwd_find_text_backbone(model):
        """定位文本 backbone:兼容普通结构(model.model.layers)与 VL
        包装结构(model.language_model[.model].layers,顶层是 visual +
        language_model)。返回 (backbone, layers),找不到返回 (None, None)。"""
        # 候选路径按优先级:普通 → VL 两级
        candidates = (
            getattr(model, "model", None),
            getattr(getattr(model, "language_model", None), "model", None),
            getattr(model, "language_model", None),
        )
        for holder in candidates:
            if holder is None:
                continue
            layers = getattr(holder, "layers", None) or getattr(
                holder, "decoder_layers", None
            )
            if layers is not None:
                return holder, layers
        return None, None

    @classmethod
    def _report_model_layers_once(cls, runner) -> None:
        """一次性打云侧实际加载的 decoder 层数与首末层号。

        半模型嫌疑的裁决依据:pp 切层错误时层数减半或首层不从 0 起
        (如 layers.14-27);正常应为全量且从 0 起。层列表取不到时打
        顶层模块名,留人工判读。"""
        if cls._layers_reported:
            return
        cls._layers_reported = True
        try:
            model = runner.get_model()
            backbone, layers = cls._lwd_find_text_backbone(model)
            if layers is not None:
                names = [
                    n for n, _ in list(backbone.named_children())
                    if "layer" in n
                ]
                cls._log(
                    "[lwd-model-dbg] decoder layers=%d (module=%s); "
                    "expect full depth starting at layer 0 — halved count "
                    "or nonzero start = PP mis-slice",
                    len(layers), names[:3],
                )
                return
            cls._log(
                "[lwd-model-dbg] top-level modules=%s",
                [n for n, _ in list(model.named_children())][:12],
            )
        except Exception:  # noqa: BLE001
            cls._log("[lwd-model-dbg] layer probe failed")

    # ------------------------------------------------------------------ #
    # cloud                                                              #
    # ------------------------------------------------------------------ #
    @classmethod
    def cloud_embeds_injected(cls, req_id, idx, start, n, buf) -> None:
        """UP embeds 注入点:写入窗口、首行数值(与 prepared 对照)。"""
        cls._log(
            "[lwd-input-dbg] inject req=%s idx=%s start=%s n=%s buf=%s "
            "row0[:6]=%s",
            req_id,
            idx,
            start,
            n,
            tuple(buf.shape),
            [round(v, 4) for v in buf[0, :6].tolist()] if buf.shape[0] else [],
        )

    @staticmethod
    def _embed_probes(runner, total: int):
        """embeds 探针对:首行前 6 值 + 全排程行整体校验和。

        emb_all 与集中式 embed_tokens hook 的 [layer-trace] 行对拍;
        行内数值即模型实际吃的输入。任一失败返回 (None, None)。"""
        try:
            emb = [
                round(v, 4)
                for v in runner.inputs_embeds.gpu[0, :6].float().cpu().tolist()
            ]
            t = runner.inputs_embeds.gpu[:total]
            emb_all = (
                f"n={t.numel()} sum={t.float().sum().item():.4f} "
                f"l2={t.float().norm().item():.4f}"
            )
            return emb, emb_all
        except Exception:  # noqa: BLE001
            return None, None

    @classmethod
    def cloud_prepared_inputs(cls, runner, num_scheduled_tokens) -> None:
        """base fill loop 之后:生效掩码/输入 id/embeds 首行/分支条件。

        判定:
        * mask 前 prompt 段应为全 False(True = 占位 id 0 被当真查表)
        * inputs_embeds[0][:6] 应与 inject 的 row0[:6] 一致(不一致 =
          fill loop 没用注入的 embeds)
        """
        # num_scheduled_tokens 是逐请求 ndarray,切片前须先求总行数,
        # 否则 min(6, ndarray) 产出数组、tensor 切片抛异常被吞成 None
        # (decode 步的 input_ids 因此一直是盲区)。
        cls._report_model_layers_once(runner)
        try:
            total = int(num_scheduled_tokens.sum())
        except Exception:  # noqa: BLE001
            total = 0
        try:
            mask = runner.is_token_ids.cpu[:total].tolist()
        except Exception:  # noqa: BLE001
            mask = None
        try:
            ids = runner.input_ids.gpu[: min(6, total)].tolist()
        except Exception:  # noqa: BLE001
            ids = None
        emb, emb_all = cls._embed_probes(runner, total)
        from vllm.distributed.parallel_state import get_pp_group

        # prefill forward 正确性三要素:positions / seq_lens / TP 组态
        try:
            p = runner.positions
            p = getattr(p, "cpu", p)  # CpuGpuBuffer 或裸 tensor 兼容
            pos = p[:num_scheduled_tokens].tolist()
        except Exception:  # noqa: BLE001
            pos = None
        try:
            from vllm.distributed.parallel_state import get_tp_group

            tp_state = (get_tp_group().world_size, get_tp_group().rank_in_group)
        except Exception:  # noqa: BLE001
            tp_state = None
        cls._log(
            "[lwd-input-dbg] prepared n=%s mask=%s input_ids[:6]=%s "
            "inputs_embeds[0][:6]=%s embeds_all{%s} first_rank=%s "
            "positions=%s tp=%s",
            num_scheduled_tokens,
            mask,
            ids,
            emb,
            emb_all,
            get_pp_group().is_first_rank,
            pos,
            tp_state,
        )

    @classmethod
    def cloud_fill_window(cls, req_id, start_pos, num_sched, window) -> None:
        """fill loop 实际拷贝点:逐请求打读自 req_prompt_embeds 的窗口摘要
        (sum/mean/head6,与 inject/边侧 SEND UP 的 DUMP 对照)。"""
        try:
            flat = window.float().reshape(-1)
            cls._log(
                "[lwd-input-dbg] fill req=%s start=%s n=%s head6=%s "
                "sum=%.4f mean=%.6f",
                req_id,
                start_pos,
                num_sched,
                [round(v, 4) for v in flat[:6].tolist()],
                float(flat.sum()),
                float(flat.mean()),
            )
        except Exception:  # noqa: BLE001
            pass

    @classmethod
    def cloud_request_admitted(cls, wire, sampling_params) -> None:
        """请求准入时打停止参数(验证 max_tokens / eos / stop_ids)。"""
        cls._log(
            "[lwd-finish-dbg] req=%s admitted: max_tokens=%s eos=%s "
            "stop_ids=%s min_tokens=%s ignore_eos=%s",
            wire.request_id,
            sampling_params.max_tokens,
            sampling_params.eos_token_id,
            sampling_params.stop_token_ids,
            sampling_params.min_tokens,
            sampling_params.ignore_eos,
        )

    @classmethod
    def cloud_step(cls, scheduler, meta, engine_core_outputs) -> None:
        """每步打 token 进度、最后一个 token 与推导出的 finish 码。"""
        from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_engine import (
            LwdCloudEngineCore,
        )

        finish_reasons = LwdCloudEngineCore._lwd_c2e_finish_reasons(
            meta, engine_core_outputs
        )
        for request_id, finish in zip(meta.req_ids, finish_reasons):
            req = scheduler.requests.get(request_id)
            if req is None:
                cls._log(
                    "[lwd-finish-dbg] req=%s finish=%s (not in scheduler)",
                    request_id,
                    finish,
                )
                continue
            cls._log(
                "[lwd-finish-dbg] req=%s out_len=%d last_tok=%s finish=%s",
                request_id,
                req.num_output_tokens,
                req.output_token_ids[-1] if req.output_token_ids else None,
                finish,
            )

    # ------------------------------------------------------------------ #
    # edge                                                               #
    # ------------------------------------------------------------------ #
    @classmethod
    def edge_tokens_delivered(
        cls, request_id, sampled_token_ids, finish_reason, vllm_config
    ) -> None:
        """边侧每拍交付的 token 解码为文字(判断乱码/通顺)。"""
        text = ""
        if sampled_token_ids:
            try:
                if cls._tokenizer is None:
                    from vllm.tokenizers.registry import get_tokenizer

                    mc = vllm_config.model_config
                    cls._tokenizer = get_tokenizer(
                        mc.tokenizer, trust_remote_code=mc.trust_remote_code
                    )
                text = cls._tokenizer.decode(sampled_token_ids)
            except Exception as e:  # noqa: BLE001
                text = f"<decode failed: {e}>"
        cls._log(
            "[lwd-token-dbg] req=%s tokens=%s text=%r finish=%s",
            request_id,
            sampled_token_ids,
            text,
            finish_reason,
        )
