import argparse
import asyncio

from tokenspeed import bench


def test_token_ids_dataset_samples_exact_lengths_without_tokenizer():
    samples = bench.sample_token_ids_requests(
        input_len=8,
        output_len=4,
        num_prompts=3,
        range_ratio=0.0,
        dataset_path=None,
        vocab_size=100,
        token_offset=10,
        prefix_len=2,
        random_seed=3,
        request_id_prefix="tok-",
    )

    assert [sample.prompt_len for sample in samples] == [8, 8, 8]
    assert [sample.expected_output_len for sample in samples] == [4, 4, 4]
    assert [sample.request_id for sample in samples] == ["tok-0", "tok-1", "tok-2"]
    assert all(isinstance(sample.prompt, list) for sample in samples)
    assert all(isinstance(token, int) for sample in samples for token in sample.prompt)
    assert all(10 <= token < 100 for sample in samples for token in sample.prompt)


def test_token_ids_dataset_varies_lengths_with_ratio():
    samples = bench.sample_token_ids_requests(
        input_len=16,
        output_len=8,
        num_prompts=16,
        range_ratio=0.5,
        dataset_path=None,
        vocab_size=200,
        token_offset=1,
        random_seed=7,
    )

    prompt_lens = {sample.prompt_len for sample in samples}
    output_lens = {sample.expected_output_len for sample in samples}

    assert min(prompt_lens) >= 8
    assert max(prompt_lens) <= 24
    assert len(prompt_lens) > 1
    assert min(output_lens) >= 4
    assert max(output_lens) <= 12
    assert len(output_lens) > 1


def test_main_async_token_ids_skips_tokenizer_init(monkeypatch):
    captured = {}

    def fail_get_tokenizer(_model_id):
        raise AssertionError("tokenizer should not be initialized for token_ids")

    async def fake_benchmark(**kwargs):
        captured.update(kwargs)
        return {
            "duration": 1.0,
            "completed": len(kwargs["input_requests"]),
            "failed": 0,
        }

    monkeypatch.setattr(bench, "get_tokenizer", fail_get_tokenizer)
    monkeypatch.setattr(bench, "benchmark", fake_benchmark)

    args = argparse.Namespace(
        append_result=False,
        apply_chat_template=False,
        backend="openai",
        base_url="http://127.0.0.1:8000",
        burstiness=1.0,
        dataset_name="token_ids",
        dataset_path=None,
        disable_ignore_eos=False,
        disable_tqdm=True,
        endpoint="/v1/completions",
        extra_body={"temperature": 0},
        extra_request_body=None,
        goodput=None,
        header=None,
        host="127.0.0.1",
        ignore_eos=False,
        input_len=8,
        insecure=False,
        label="unit",
        logprobs=None,
        max_concurrency=2,
        max_model_len=None,
        metric_percentiles="99",
        model="unit-model",
        num_prompts=2,
        num_warmups=0,
        output_file=None,
        output_len=4,
        percentile_metrics=None,
        port=8000,
        profile=False,
        profile_num_steps=None,
        ramp_up_end_rps=None,
        ramp_up_start_rps=None,
        ramp_up_strategy=None,
        random_input_len=1024,
        random_output_len=128,
        random_prefix_len=0,
        random_range_ratio=0.0,
        ready_check_timeout_sec=0,
        request_id_prefix="bench-",
        request_rate=float("inf"),
        result_dir=None,
        save_detailed=False,
        save_result=False,
        seed=0,
        served_model_name=None,
        sharegpt_output_len=None,
        skip_min_tokens_check=False,
        skip_tokenizer_init=False,
        tokenizer=None,
        token_ids_input_len=1024,
        token_ids_output_len=128,
        token_ids_prefix_len=0,
        token_ids_range_ratio=0.0,
        token_ids_token_offset=1,
        token_ids_vocab_size=32000,
        trust_remote_code=True,
    )

    result = asyncio.run(bench.main_async(args))

    assert result["tokenizer_id"] is None
    assert captured["tokenizer"] is None
    assert captured["ignore_eos"] is True
    assert [request.prompt_len for request in captured["input_requests"]] == [8, 8]
    assert [request.expected_output_len for request in captured["input_requests"]] == [
        4,
        4,
    ]
    assert all(
        isinstance(token, int)
        for request in captured["input_requests"]
        for token in request.prompt
    )
