from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.engine.generation_output_processor import (
    OutputProcesser,
    RequestState,
)
from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
from tokenspeed.runtime.metrics.collector import EngineMetrics
from tokenspeed.runtime.sampling.sampling_params import SamplingParams
from tokenspeed.runtime.sampling.utils import write_output_logprobs


def test_write_output_logprobs_attaches_per_row_topk() -> None:
    logits = torch.tensor(
        [
            [1.0, 3.0, 2.0, 0.0],
            [0.0, 1.0, 4.0, 2.0],
            [2.0, 1.0, 0.0, 3.0],
        ]
    )
    tokens = torch.tensor([1, 2, 3], dtype=torch.int32)
    output = LogitsProcessorOutput(next_token_logits=logits)

    write_output_logprobs(output, logits, tokens, top_logprobs_nums=[2, 0, 3])

    raw_logprobs = torch.log_softmax(logits, dim=-1)
    assert torch.allclose(
        output.next_token_logprobs,
        raw_logprobs.gather(-1, tokens.long().unsqueeze(-1)).squeeze(-1),
    )
    assert output.next_token_top_logprobs_nums == [2, 0, 3]
    assert torch.allclose(
        output.next_token_top_logprobs_val,
        raw_logprobs.topk(3, dim=-1).values,
    )
    assert torch.equal(
        output.next_token_top_logprobs_idx,
        raw_logprobs.topk(3, dim=-1).indices,
    )


def test_write_output_logprobs_repeats_request_topk_for_verify_rows() -> None:
    logits = torch.tensor(
        [
            [2.0, 1.0, 0.0],
            [1.0, 3.0, 0.0],
            [0.0, 1.0, 4.0],
            [3.0, 2.0, 1.0],
        ]
    )
    output = LogitsProcessorOutput(next_token_logits=logits)

    write_output_logprobs(
        output,
        logits,
        torch.tensor([0, 1, 2, 0], dtype=torch.int32),
        top_logprobs_nums=[1, 2],
    )

    assert output.next_token_top_logprobs_nums == [1, 1, 2, 2]


class _CaptureSender:
    def __init__(self) -> None:
        self.sent = []

    def send_pyobj(self, obj) -> None:
        self.sent.append(obj)


class _ForwardOp:
    request_ids = ["rid-1"]
    input_lengths = [1]
    extend_prefix_lens = []
    request_pool_indices = [0]

    def num_extends(self) -> int:
        return 0


class _ExecutionResult:
    output_tokens = torch.tensor([10, 20], dtype=torch.int32)
    output_lengths = torch.tensor([2], dtype=torch.int32)
    output_logprobs = torch.tensor([-0.5, -0.6])
    output_top_logprobs_val = torch.tensor([[-0.1, -0.2], [-0.3, -0.4]])
    output_top_logprobs_idx = torch.tensor([[10, 11], [20, 21]], dtype=torch.int64)
    output_top_logprobs_nums = [2, 1]
    grammar_completion = None

    def sync(self) -> None:
        return None


def _sampling_params(**kwargs) -> SamplingParams:
    params = SamplingParams(**kwargs)
    params.normalize(None)
    return params


def test_generation_output_processor_sends_output_top_logprobs() -> None:
    sender = _CaptureSender()
    processor = OutputProcesser(
        send_to_tokenizer=sender,
        metrics=EngineMetrics(labels={}, enabled=False),
    )
    processor.register(
        "rid-1",
        RequestState(
            prompt_input_ids=[1],
            sampling_params=_sampling_params(max_new_tokens=2),
            stream=False,
            tokenizer=None,
            return_logprob=True,
            top_logprobs_num=2,
        ),
    )

    processor.post_process_forward_op(_ForwardOp(), _ExecutionResult())

    assert len(sender.sent) == 1
    batch = sender.sent[0]
    assert batch.output_token_logprobs_val[0] == pytest.approx([-0.5, -0.6])
    assert batch.output_top_logprobs_val[0][0] == pytest.approx([-0.1, -0.2])
    assert batch.output_top_logprobs_val[0][1] == pytest.approx([-0.3])
    assert batch.output_top_logprobs_idx == [[[10, 11], [20]]]
