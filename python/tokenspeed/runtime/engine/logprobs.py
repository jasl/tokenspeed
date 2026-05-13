# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Logprob detokenization for the async frontend.

A dedicated processor turns sampler-produced logprob arrays
(``recv_obj.{input,output}_{token,top}_logprobs_{val,idx}``) into the
``logprobs_info`` payload the per-request ``RequestOutputCollector``
merges. Lives next to ``OutputProcessor`` rather than inside it so
F.2's empty-logprob root-cause fix has a single, isolated home.

The engine reference lets us read the live ``tokenizer`` (mutated on
``update_weights_from_disk``) without snapshotting it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tokenspeed.runtime.engine.io_struct import BatchStrOut

if TYPE_CHECKING:
    from tokenspeed.runtime.engine.async_llm import AsyncLLM


class LogprobsProcessor:
    """Translate sampler logprob arrays into per-request meta_info entries.

    Holds an engine reference solely for the live ``tokenizer`` it
    needs in ``detokenize_logprob_tokens`` when the caller asks for
    text decoding (``return_text_in_logprobs=True``). When the caller
    sets ``return_text_in_logprobs=False`` the tokenizer is never
    touched, which is the mode the stub-tokenizer test paths exercise.
    """

    def __init__(self, engine: AsyncLLM) -> None:
        self.engine = engine

    def convert_logprob_style(
        self,
        logprobs_info: dict,
        top_logprobs_num: int,
        token_ids_logprob: list[int],
        return_text_in_logprobs: bool,
        recv_obj: BatchStrOut,
        recv_obj_index: int,
    ) -> None:
        # Defensive: sampler may not have populated logprobs for this request
        # (e.g. backend doesn't support logprobs, overlap race). Return empty
        # slots rather than crashing the whole AsyncLLM loop.
        def _get(field: str):
            lst = getattr(recv_obj, field, None) or []
            if recv_obj_index < len(lst):
                return lst[recv_obj_index]
            return []

        input_token_logprobs = logprobs_info.get("input_token_logprobs", [])
        output_token_logprobs = logprobs_info.get("output_token_logprobs", [])

        input_token_logprobs.extend(
            self.detokenize_logprob_tokens(
                _get("input_token_logprobs_val"),
                _get("input_token_logprobs_idx"),
                return_text_in_logprobs,
            )
        )

        output_token_logprobs.extend(
            self.detokenize_logprob_tokens(
                _get("output_token_logprobs_val"),
                _get("output_token_logprobs_idx"),
                return_text_in_logprobs,
            )
        )

        logprobs_info["input_token_logprobs"] = input_token_logprobs
        logprobs_info["output_token_logprobs"] = output_token_logprobs

        if top_logprobs_num > 0:
            input_top_logprobs = logprobs_info.get("input_top_logprobs", [])
            output_top_logprobs = logprobs_info.get("output_top_logprobs", [])

            input_top_logprobs.extend(
                self.detokenize_top_logprobs_tokens(
                    _get("input_top_logprobs_val"),
                    _get("input_top_logprobs_idx"),
                    return_text_in_logprobs,
                )
            )
            output_top_logprobs.extend(
                self.detokenize_top_logprobs_tokens(
                    _get("output_top_logprobs_val"),
                    _get("output_top_logprobs_idx"),
                    return_text_in_logprobs,
                )
            )

            logprobs_info["input_top_logprobs"] = input_top_logprobs
            logprobs_info["output_top_logprobs"] = output_top_logprobs

        if token_ids_logprob is not None:
            input_token_ids_logprobs = logprobs_info.get("input_token_ids_logprobs", [])
            output_token_ids_logprobs = logprobs_info.get(
                "output_token_ids_logprobs", []
            )

            input_token_ids_logprobs.extend(
                self.detokenize_top_logprobs_tokens(
                    _get("input_token_ids_logprobs_val"),
                    _get("input_token_ids_logprobs_idx"),
                    return_text_in_logprobs,
                )
            )
            output_token_ids_logprobs.extend(
                self.detokenize_top_logprobs_tokens(
                    _get("output_token_ids_logprobs_val"),
                    _get("output_token_ids_logprobs_idx"),
                    return_text_in_logprobs,
                )
            )

            logprobs_info["input_token_ids_logprobs"] = input_token_ids_logprobs
            logprobs_info["output_token_ids_logprobs"] = output_token_ids_logprobs

    def detokenize_logprob_tokens(
        self,
        token_logprobs_val: list[float],
        token_logprobs_idx: list[int],
        decode_to_text: bool,
    ):
        if not decode_to_text:
            return [
                (logprob, token_id, None)
                for logprob, token_id in zip(token_logprobs_val, token_logprobs_idx)
            ]
        else:
            assert self.engine.tokenizer is not None
            token_texts = self.engine.tokenizer.batch_decode(
                [[token_id] for token_id in token_logprobs_idx]
            )
            return list(zip(token_logprobs_val, token_logprobs_idx, token_texts))

    def detokenize_top_logprobs_tokens(
        self,
        token_logprobs_val: list[float],
        token_logprobs_idx: list[int],
        decode_to_text: bool,
    ):
        # We should batch all top-k tokens in all positions.
        ret = []
        for logprobs, token_ids in zip(token_logprobs_val, token_logprobs_idx):
            if logprobs:
                ret.append(
                    self.detokenize_logprob_tokens(logprobs, token_ids, decode_to_text)
                )
            else:
                ret.append(None)
        return ret
