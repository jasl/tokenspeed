# Benchmarking

TokenSpeed includes an online serving benchmark client:

```bash
tokenspeed bench serve \
  --backend openai \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --dataset-name random \
  --input-len 1024 \
  --output-len 1024 \
  --num-prompts 2 \
  --max-concurrency 2 \
  --extra-body '{"temperature": 0}'
```

The default `random` dataset initializes the model tokenizer, generates token
ids, decodes them to text, then sends text prompts. This gives text-shaped
requests, but it requires the local benchmark environment to support the model's
tokenizer and config.

For new architectures whose tokenizer is not yet available in local
Transformers, use the `token_ids` dataset. It sends integer token ids directly
to the OpenAI-compatible completions endpoint and skips tokenizer
initialization:

```bash
tokenspeed bench serve \
  --backend openai \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --dataset-name token_ids \
  --input-len 1024 \
  --output-len 1024 \
  --num-prompts 2 \
  --max-concurrency 2 \
  --request-rate inf \
  --extra-body '{"temperature": 0}'
```

`--input-len` and `--output-len` map to `--token-ids-input-len` and
`--token-ids-output-len` for this dataset. Use `--token-ids-vocab-size` and
`--token-ids-token-offset` if the target model needs a different valid token-id
range. `--token-ids-prefix-len` creates a shared token prefix while preserving
the configured total input length.
