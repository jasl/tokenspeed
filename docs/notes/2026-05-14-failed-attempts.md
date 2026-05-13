# Failed Attempts Log

Catalog of optimisation attempts that were tried, measured, and reverted.
The intent is to keep negative results visible so the next round doesn't
re-discover them from scratch.

## 2026-05-14 — V3 per-type planner cache (skipped, not applicable)

* **Hypothesis** (vLLM `sparse_swa.py:185-191` pattern): cache the FlashMLA
  planner state per ``(kind, batch_size_bucket, total_seq_len_bucket)`` so
  the 20 HCA + 21 CSA layers re-use the plan instead of re-running the
  per-layer planner.
* **Audit finding**: TokenSpeed already caches ``_get_decode_tile_metadata``
  per ``(phase, kind, bs)`` (``python/tokenspeed/runtime/layers/attention/
  backends/deepseek_v4.py:613``). The deeper SWA decode metadata
  (``_get_decode_swa_metadata``) is keyed on ``(window_size, block_size,
  positions.numel())``, which would benefit from a multi-entry cache — but
  ``swa_window`` is read directly from ``config.sliding_window`` (line
  2046 of ``deepseek_v4.py``) and is the **same single value across every
  layer of every kind**. The current single-entry cache already hits on the
  2nd+ layer of every step.
* **Verdict**: Not applicable to our model config. No change shipped.
* **Where to revisit**: If a future DSv4 variant exposes per-kind windows,
  re-evaluate.

## 2026-05-14 — D1 multi-row register tile for `gate_activation_warp_ds4_decode_kernel` (-14.7%, reverted)

* **Hypothesis**: each warp accumulates 4 (gate, up) row pairs sharing the
  same (token, choice). Activation read `x = hidden_states[token, k]` is
  loaded once per `k` and reused across 8 inner FMAs (4 gate + 4 up).
  Total work units shrink 4×; 188 SMs × 6 warps/SM should still saturate.
* **Implementation**: `gates[4]`/`ups[4]` register arrays inside the warp,
  per-row offsets computed up-front, ``#pragma unroll`` on the inner row
  loop. Grid sized over row-groups, not individual rows.
* **Build / occupancy**: ``cuobjdump --dump-resource-usage`` reported 40
  REG (vs 34 baseline, +6 regs), STACK=0, no spills, 6 blocks/SM still
  achievable.
* **Correctness**: ``test_sm12x_mxfp4.py`` passed 34/34.
* **Bench (SM120 TP=2, token_ids 1024×1024 c=1, graph mode)**:
  21.70 baseline → 22.07 V2 Stage 1 → **18.83 multi-row** (`-14.7%` vs
  Stage 1, TPOT 43.0 ms → 50.8 ms).
* **Root cause**: parallelism collapsed. 12288 rows / 8 warps-per-block
  was 1536 blocks (8.2 blocks/SM); after the 4× shrink the grid was 384
  blocks (~2.04 blocks/SM). With per-warp work also 4× heavier and an
  8-wide register-dependent FMA chain inside the k loop, the SMs were
  under-occupied and the longer inner loop blew the latency budget. The
  memory note ``feedback_blackwell_parallelism.md`` already warned about
  exactly this on the 140-SM number; SM120 actually has 188 SMs, making
  the problem worse, not better.
* **Verdict**: reverted to single-row-per-warp form (commit-clean).

## 2026-05-14 — D1 shared-memory activation cache for `gate_activation_warp_ds4_decode_kernel` (≈0%, reverted)

* **Hypothesis**: the 8 warps in a block read the same
  ``hidden_states[token, 0..4095]`` (kDs4Intermediate=2048 divides
  kWarpsPerBlock=8 so a block stays inside one (token, choice)). Prefetch
  the 8 KB bf16 activation slab into ``__shared__`` once per block; all
  warps then read from shmem instead of L2.
* **Implementation**: new ``gate_activation_warp_ds4_decode_smem_kernel``
  + dispatcher gate when grid fits in one iteration. Cooperative
  ``__syncthreads()``-guarded shmem load at block entry, then the
  per-warp dot product reads from ``s_hidden[k]``.
* **Build / occupancy**: REG=32 (lower; no stride loop), SHARED=9216 (≈8
  KB activation buffer + alignment), STACK=0. 6 blocks/SM × 9 KB = 54 KB
  fits in the 102 KB per-SM shmem budget.
* **Correctness**: ``test_sm12x_mxfp4.py`` passed 34/34.
* **Bench**: 22.07 Stage 1 baseline → **21.98 shmem** (-0.4%, TPOT 43.20
  vs 43.03 ms). Within run-to-run noise.
* **Root cause**: SM120 has a **134 MB L2 cache** (cf. the ~50 MB on
  SM100). An 8 KB activation slab is trivially L2-resident after the
  first read, so the 8 sibling warps in a block already hit L2 at a
  fraction of HBM cost. Replacing those L2 hits with shmem reads removes
  one cache level worth of bandwidth, but the explicit cooperative load
  + ``__syncthreads()`` introduces a sequential bottleneck and an extra
  write into shmem (HBM → shmem → registers vs HBM → L2 → registers).
  The two roughly cancel.
* **Verdict**: reverted. shmem caching is only a win when the working
  set significantly exceeds L2 or when the activation buffer is read so
  many times that L2 line eviction becomes a problem — neither holds
  here at bs=1 with 4096-element activations.

## Cross-cutting lesson

For bs=1 decode on SM120, the existing MoE warp kernels
(`gate_activation_warp_ds4_decode_kernel`,
`down_warp_ds4_decode_kernel`) sit close to the bandwidth ceiling.
Both of the canonical bandwidth-reduction tricks (register tiling,
shmem caching) hit the wall:

* Register tiling drops the block count below the SM occupancy needed
  to hide global-memory latency on a 188-SM device.
* Shmem caching cannot beat a 134 MB L2 that already absorbs the
  duplicate reads.

Future MoE wins should pursue **structural** changes — fusing W13 + W2
to drop the intermediate write/read pair, replacing the two-launch
sequence with a single grid-sync kernel, or moving from the warp path
to a properly tiled tensorcore path at the M~6 shape — rather than
chasing more bandwidth tricks inside the current warp kernel.
