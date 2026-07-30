# tomsanbear/candle — fork bill of materials

The **incubation trunk** for candle changes born in downstream production use: an on-device streaming TTS app, a PDF→Markdown document-conversion pipeline, and a local-LLM inference runner for ternary-quantized models — all Metal-first on Apple Silicon. Changes land on `tomsanbear-dev`, prove themselves in production, then graduate to focused `fix/*` / `feat/*` / `perf/*` branches and upstream PRs. This file is the single inventory: every carried change, its upstream status, and our confidence in it.

**Base**: `tomsanbear-dev` is merged with upstream `main` at 0.11.0 (`31f35b14`, merge commit `23b9460e`, 2026-07-22). Validation at the merge: candle-metal-kernels 60/60, candle-core `--features metal` 334/334, candle-nn `--features metal` 75/75.

**Status**: `Incubating` (on dev, not yet PR'd) · `Submitted (#PR)` · `Merged (#PR)` · `Rejected (#PR)` · `Refuted` (we measured it and killed it ourselves) · `Superseded` (upstream solved it another way) · `Wanted` (need identified, nothing written).

**Confidence**: `proven` (shipping downstream with receipts) · `tested` (unit/bench coverage, not in production) · `experimental` · `n/a`.

**Maintenance protocol** (humans and agents): add an entry when a change lands on `tomsanbear-dev`; move it when a PR opens, merges, or closes. Never delete Rejected/Refuted/Superseded entries — their receipts are what stop the idea being re-litigated. One entry per logical change, not per commit. Record a need in [Wanted](#wanted) when it is identified but unwritten, and check [Upstream coordination](#upstream-coordination) before opening any PR.

## Contents

- [Upstream coordination](#upstream-coordination) — maintainer plans that overlap with carried work; read before PRing
- [In flight](#in-flight) — open upstream PRs
- [Incubating on `tomsanbear-dev`](#incubating-on-tomsanbear-dev) — landed on dev, not yet PR'd
- [Incubating on side branches](#incubating-on-side-branches) — larger work not yet on dev
- [Merged upstream](#merged-upstream) — this fork's landed history
- [Closed with receipts](#closed-with-receipts) — rejected, refuted, and superseded, kept so nobody retries them blind
- [Wanted](#wanted) — needs identified, nothing written yet
- [Open investigations](#open-investigations) · [Branch hygiene](#branch-hygiene)

## Upstream coordination

Maintainer work in flight that overlaps what this fork carries, verified 2026-07-23. Check here before opening a PR — several of these want a conversation rather than a patch.

- **Norm kernels** — #3150 (a third-party Metal layernorm PR) was closed by ivarflakstad: "we already have metal layer norm with almost the exact same performance characteristics… I was planning on reusing some of the concepts there [reduce / arg reduce / softmax] in a revamp of the layer and rms normalization kernels." So a norm-kernel revamp is *planned upstream*, and a same-mechanism PR has already been turned away once. Our batched norms differ in mechanism (simdgroup-per-row batching, vec4 loads, counter-driven threadgroup geometry), but the right first move is offering the counter data to that revamp, not a PR.
- **#3188 Generic KvCache** (open since 2025-11) — our `truncate_to` should probably land on top of this rather than against today's cache.
- **#3496 Backend-driven sampling and repeat penalty** (open) — a device-resident top-k is a natural fit here rather than as a standalone op.
- **#3467 Lazy backend** (open) — the structural answer to dispatch overhead, which is what our `compute_per_buffer` knob and profiling work address tactically. Understand its direction before pitching either.
- **#2906 Add metal precompilation** (open since 2025-04) — same pipeline-cache surface as #3760/#3761.
- Recent maintainer throughput is heavily CPU-side (neon dotprod, threadpools, quantized CPU perf: #3570–#3658), so the Metal surface is comparatively uncontested right now.

## In flight

Open upstream PRs. All the 2026 ones are also carried on `tomsanbear-dev` (cherry-picked 2026-07-22, re-validated on the 0.11 base). No review activity on any of the four open PRs as of 2026-07-23.

### Shared locks for Metal pipeline-cache hits
`metal-core` · `perf/metal-cache-read-locks` · Submitted (#3761) · tested

The pipeline cache is read-mostly; taking the exclusive lock on every hit was a contention point. Hits now take a shared read lock. Touches the same surface as the maintainer's open #2906.

### CPU batched matmul with broadcast (stride-0) views
`cpu` · `fix/cpu-stride-zero-batched-matmul` · Submitted (#3758) · tested

### Empty binary-op validation and gradients
`core/autograd` · `fix/empty-binary-validation-autograd` · Submitted (#3757) · tested

### Zero-sized matmul validation and gradients
`core/autograd` · `fix/zero-matmul-validation-autograd` · Submitted (#3756) · tested

Dev carries the 3-arg helper as `assert_zero_grad_shaped` (renamed at cherry-pick to coexist with #3757's 2-arg one).

### Legacy 2024 Metal PRs — triage
`metal-core` · (legacy branches) · Submitted (#2061, #2037) · n/a

#2061 makes Metal kernels take an encoder rather than a command buffer; #2037 is a command encoder/buffer reuse refactor. Both are 2024-era and superseded by later upstream work. Close them or refresh — they are dead weight on the maintainer's queue either way.

### Rust 1.97 clippy fixes and LogitsProcessor NaN guard
`workspace` · third-party #3754 (GregoryBolshakov) · **Merged upstream 2026-07-23** · tested

Cherry-picked (`-x`) onto dev 2026-07-22 with authorship preserved, while it was still open; also clears the long-standing shape.rs warning. Now landed upstream, so the carry is redundant and drops out at the next upstream merge — expect a cherry-pick collision there and resolve in upstream's favour.

## Incubating on `tomsanbear-dev`

### Operation-local seeded random tensors
`core` / `metal-kernels` · `99ba7b04` · Incubating · **proven — deterministic and 3.5–6.9× faster on M3**

`Tensor::rand_seeded` and `Tensor::randn_seeded` give one operation an immutable `u64` seed without advancing the device's stateful RNG. The old downstream workaround — `Device::set_seed` followed by an asynchronously queued Metal random kernel — was not deterministic: a later host reset could overwrite the one shared seed buffer before an earlier kernel read it, and thread zero mutated that buffer while sibling threads were still loading it. CPU now owns a call-local `StdRng`; Metal copies the seed into command-owned parameter bytes and maps logical groups of four outputs through full-key Philox4x32-10 counters. Same-backend/build results are exact and prefix-stable; cross-backend identity is not promised. CUDA is explicitly unsupported until a native stateless kernel exists. Metal intentionally exposes F32 only: converting f32 uniforms near the exclusive upper bound to F16/BF16 can round to the bound, so advertising half support would violate `[lo, up)`.

M3 Criterion medians at 50,240 / 998,720 / 9,026,880 elements: explicit Metal is 3.5× / 6.5–6.9× / 5.2× faster than the mutable stateful path; host generation plus upload is 40–99× / 151–369× / 129–335× slower than explicit Metal. The benchmark queues each tensor and synchronizes the batch; it does not read tensors back. Full `candle-core --features metal`, Metal-kernel, and warnings-denied clippy gates pass; the CUDA unsupported-boundary test passes on the CUDA host. Compiling mutations proved queued-seed independence, cloned-handle concurrency, upper-32-bit sensitivity, all four published Philox words and ten rounds, uniform and normal prefix stability, bounds/moments, validation semantics, half/CUDA backend boundaries, and stateful RNG isolation.

### Fused SwiGLU kernel
`metal-kernels` / `candle-nn` · `4e892d30` · Incubating · tested (`swiglu` cpu+metal)

`candle_nn::ops::swiglu` was three lines of Rust — `chunk(2, last)` → `silu` → `mul` — with **no kernel behind it**, which is easy to mistake for a fused op when scanning the `ops::` surface. It now dispatches a real kernel; `swiglu_slow` keeps the composed spelling as the reference.

Why it matters beyond the two saved dispatches: it is what makes a **width-fused gate|up projection** pay. One wide GEMM instead of two is arithmetically free, but the composed tail then reads both halves as *strided* views and materializes the `silu` result (3 reads + 2 writes over 2 dispatches vs 2 reads + 1 write over 1). Measured downstream on an M3 Pro (30-layer 1024-wide trunk), width-fusing gate|up **with the composed tail was a loss at every sequence length**: +0.9% at seq=1, +2.2% at 16, +13% at 64, +17% at 256. That is the disproof this kernel exists to flip.

`silu` is evaluated through the same `usilu` functor the standalone `silu` kernel instantiates, so fused and composed agree bit-for-bit on the activation. Gotcha for anyone extending it: `half` is a Metal type keyword and cannot name a parameter.

### Fused residual-add + RMSNorm
`metal-kernels` / `candle-nn` · `4e892d30` · Incubating · tested (`add_rms_norm` cpu+metal) — **measured neutral downstream, see the receipt below**

`candle_nn::ops::add_rms_norm(xs, residual, alpha, eps) -> (sum, normed)`. A transformer block always needs both halves of `x = x + sublayer; h = rms_norm(x)` — the sum continues the residual chain, the normalization feeds the next sublayer — so writing them separately is two dispatches and two passes over the row.

**The interesting part is the API shape.** candle's `CustomOpN` returns exactly one storage and `InplaceOpN` returns none, so a two-output kernel looks inexpressible without either an API change or mutating an input's buffer (unsafe under candle's shared-storage tensors). It is expressible: pack both outputs into one `(2, rows, cols)` buffer. Because the pack dim is **leading**, `i(0)`/`i(1)` are contiguous slices, so both come back as ordinary contiguous tensors at no copy. A trailing pack — the obvious first instinct — would force strided reads and hand the saving straight back; the test asserts contiguity so that regression cannot land silently.

One threadgroup per row with a two-level (simd, then cross-simdgroup) reduction, deliberately *not* the simdgroup-per-row `*_batched` geometry, which regresses the 1–2 row decode case. The row sum is rounded to `T` before the sum-of-squares accumulates it, matching what the composed pair does, so the paths agree at bf16/f16 rather than only at f32. Non-Metal, non-contiguous, and dtype-mismatched callers fall back to `add_rms_norm_slow`.

**RECEIPT — it did not pay off downstream, and the reason is instructive.** Wired into a 30-layer 1024-wide trunk on an M3 Pro and A/B'd against the composed pair, it measured **neutral to slightly negative**: +0.7% at seq=16, +0.9% at seq=1, +0.3% at 64, within noise at 256. The dispatch-count argument (one kernel instead of two) predicted ~2–4% and was simply wrong, because it ignored *what the second dispatch was*: the stock `rms_norm` this replaces is not naive — it has vec4 loads and switches to the simdgroup-per-row batched kernel above 32 rows. Trading a well-optimized norm plus a cheap elementwise add for one unoptimized fused pass is a wash. Saving a dispatch only wins when the dispatch you fold in was not already carrying the optimizations you lose.

Kept rather than refuted: the op is correct, tested, and the leading-dim packing is the reusable finding. It should get the vec4 body and a batched variant before anyone re-measures it — until then, do not reach for it expecting a win. (The sibling `swiglu` in the same commit is the opposite case: the composed path there had *no* kernel at all, and it delivered the full −6.0% at seq=16.)

### `rms_norm` accepts a weight dtype ≠ the activation dtype
`candle-nn` · `d60d9e4d` · Incubating · tested (`rms_norm_mixed_dtype` cpu+metal)

The fused `rms_norm` kernels (cpu/cuda/metal) are same-dtype — input and weight must match — so an F32 norm weight (the default GGUF `dequantize`) applied to F16 activations bailed with "rmsnorm is not implemented for F16 F32". Aligns the weight to the activation dtype inside `rms_norm`/`rms_norm_slow`; the result is the activation dtype, so no weight precision that would survive the output is lost, and same-dtype callers are untouched. Surfaced running an in-process Qwen3-0.6B decoder in native F16 (halves activation + KV traffic vs the F32-embedding path). Small, self-contained upstream-PR candidate.

### conv im2col paths discarded the contiguous kernel copy
`core` (all 3 backends) · `14ae55a3` · Incubating · tested (regression across cpu/metal/cuda)

Pre-existing upstream bug: non-contiguous conv kernels read garbage. Smallest review surface of anything here — good first upstream PR.

### Metal quantized matvec tail out-of-bounds and dispatch geometry
`metal-quantized` · `d2318473` · Incubating · tested (`quantized_matmul_mv_*_metal` on 0.11)

Out-of-bounds read on tail rows, plus a dispatch-geometry fix. Branch `fix/metal-qmv-tail` is ready to submit; **no upstream PR yet**.

### Cast-matrix coverage and the generic `to_dtype` sweep
`metal-kernels` / `cuda-kernels` / `core` · `6b02745d`, `13adaa2a`, `d6a3585a`, `7a4c7910` · Incubating · tested

The suite's first *generic* `to_dtype` device sweep, plus the holes it found. The sweep is worth more than any individual fix and should lead the PR. Three separable fixes:

- **Metal i32/i16** (`6b02745d`) — filled the i32/i16 rows and columns of the cast matrix. Kills the downstream workaround class where token-id tensors round-trip to the CPU purely to widen i32→i64 for an embedding `index_select`.
- **CUDA i64↔f16/bf16** (`13adaa2a`) — the sweep hit `CUDA_ERROR_NOT_FOUND` for `cast_bf16_i64`: the token-ids ↔ half-precision class had no kernels at all. CUDA still lacks i16/i32 casts (skipped in the sweep).
- **Metal f64 via software bit-conversion** (`d6a3585a`) — MSL has no double at any feature level (spec §2.1, re-verified including the feature tables), but casts don't need double *arithmetic*: f64 buffers are read/written as `ulong` and converted in software (RNE f64→f32, exact f32→f64, exact int↔f64 to 2^53 with Rust-`as` saturation). Verified for RNE ties, specials, saturation, and 2^40+ exactness; the sweep now runs the full 9×9 matrix on Metal, contiguous and strided. Unblocked a downstream P0 ("Metal contiguous to_dtype F64 F16/F32 not implemented" from host-side f64 scalars). CUDA casts f64 natively and joined the sweep in `7a4c7910`.

### Wide-row top-k kernel and wide `arg_sort` routing
`metal-kernels` / `cuda-kernels` / `core` / `candle-nn` · `2c6e5abd` + CUDA follow-up · Incubating · **proven** (Metal M3 Pro; CUDA RTX 4070)

M3 Pro, isolated: RT-DETR-style query selection (8×24000, k=300) runs 1.04 ms on device vs 3.09 ms for the host round-trip (3.0×), and the result stays on-device — no mid-graph sync, which is the structural win.

**CUDA (2026-07-23):** `candle_nn::ops::topk` used to compose `arg_sort_last_dim` on CUDA. That path sizes dynamic shared memory as `ncols_pad * 4` bytes, so Heron RT-DETR encoder rows (≈8400 tokens → pad 16384 → **64 KiB**) exceeded the default 48 KiB block limit and launched as `CUDA_ERROR_INVALID_VALUE`. Fix: a CUDA **tile-and-merge topk** kernel in `candle-kernels/src/sort.cu` (same design as Metal: shared O(TILE+k), not O(ncols)), wired through `TopK::cuda_fwd` and selected for CUDA in `ops::topk` for F32/F16/BF16 with `k_pad ≤ 1024`. Wide full `arg_sort` on CUDA now **opts in** dynamic shared up to ~99 KiB and **bails with an explicit error** beyond that (use `topk` when only k ≪ ncols). Tests: `candle-nn` `topk` covers 4096 / **8400** / **24000** last-dims at k=300 on cpu/cuda/metal.

Metal half (unchanged): `arg_sort_last_dim` beyond 1024 columns routes ascending to MLX multi-block kernels; wide descending bails clearly. The top-k kernel itself is tile-and-merge (bitonic tile sort, then a log2(2k)-stage bitonic merge with a descending carry), k padded to a power of two, f32/f16/bf16. At 8 rows it underfills the GPU (8 threadgroups) — multi-threadgroup-per-row is the optimization if a workload ever needs it. See [Upstream coordination](#upstream-coordination) re #3496.

### conv_transpose1d padded col2im fast path
`core` (all 3 backends) · `70452db0` · Incubating · **proven — 41–191× on M3 Pro**

Upsampler stages of a HiFi-GAN-style vocoder, f32 and bf16 alike: 512ch k16s8 78.9 ms → 441 µs; 68.8 ms → 688 µs; 35.3 ms → 857 µs. A tensor-level centre-crop rewrite of the padded col2im shim unblocks the existing fast path, which upstream gates on `padding == 0`. PyTorch-referenced padded/output-padded/groups tests added (zero-padded coverage already existed). Supersedes the model-side GEMM-based conv-transpose workaround the downstream TTS vocoder carries, once that app repins. Prime upstream-PR candidate.

### Fused GEMM/GEMV bias epilogue
`metal-kernels` / `candle-nn` · `2ee72f8e`, `0b182601` · Incubating · **proven — 2.0× prefill on M3 Pro**

`candle_nn::ops::matmul_bias`. Prefill 355×512×512: 139.9→69.7 µs f32, 158.6→79.3 µs bf16. Decode 1×1024×4096 is weight-bandwidth bound (~1–3%). Almost entirely Rust wiring — the vendored MLX kernels already ship `_axpby1` gemv variants and the steel `use_out_source`/`do_axpby` epilogue, they were just never exposed. Gotcha: buffer-7 batch strides grow a third C segment when `use_out_source` is set. `Linear::forward` is unchanged (explicit opt-in API).

### Fused GEMM/GEMV activation epilogue
`metal-kernels` / `candle-nn` · `d37a2fa5` · Incubating · **proven — activation is free**

`candle_nn::ops::matmul_bias_act` (relu/gelu/silu). Prefill 355×512×512: fused silu 69.7 µs ≈ plain `matmul_bias` 69.9 µs, vs 73.0 µs with a separate silu dispatch (f32; bf16 84.4→81.7). Decode 1×1024×4096 weight-bandwidth-bound (~1–2%).

Function constant 120 (`ushort` activation kind) in mlx_gemm.metal and gemv.metal; formulas mirror unary.metal's urelu/ugelu/usilu and evaluate in the output type, so the fused path matches the composed chain's precision domain — verified at 1e-5 on-device across every steel store branch and both gemv kernels, with and without bias. `act == 0` folds to identity at pipeline specialization. The candle-metal-kernels layer already supports activation without bias; a no-bias `matmul_act` surface in candle-nn is a trivial follow-up. Bench note: full 16-group sweeps produced one thermal artifact (composed f32 prefill 457 µs in-sweep vs 141 µs isolated) — receipts above use isolated readings.

### Batched simdgroup-per-row layernorm/rmsnorm
`metal-kernels` · `8e9f701a` · Incubating · **proven — 1.3–2.7× on M3 Pro**

layer_norm 1024×512: 2.5–2.7× (43.9→17.6 µs f32). rms 1024×512: 1.5–1.6×; 1024×1024: 1.3–2.3×. rows=1 is flat — the rows>=32 routing heuristic keeps decode shapes on the stock kernels. Ported from the downstream TTS app's private fused-norm kernels. The former known issue (rms f32 wide rows −11% vs stock) is resolved in-kernel by the next entry.

**Do not PR without talking to the maintainer first** — see [Upstream coordination](#upstream-coordination): a norm revamp is planned upstream and a same-shaped PR (#3150) was already closed as redundant.

### Wide-row fix for the batched norms
`metal-kernels` · `e8f52701` · Incubating · **proven — isolated groups on M3 Pro**

rms f32 1024×1024: 45.2→37.2 µs (−18%, flipping −11%-vs-stock to roughly +8%). rms f32 1024×512: 17.2→15.4 µs. rms bf16 1024×1024: 25.7→12.4 µs (−52%). layer_norm f32 1024×1024: 46.1→41.1 µs. Decode rows flat (stock path untouched, 2.64 µs).

Root-caused with gpudebug counters (M3 Pro, macOS 27, gpudebug 1.0): 32-row 1024-thread groups reached 45% occupancy against an 86% manager target, with L1 evictions at 0 and the launch limiter at 1% — occupancy quantization plus latency-exposed scalar load chains, **not** L1 thrash (that hypothesis was refuted; stock's costs are an 80–85% launch limiter and L1 evictions of 25). Fix: a vec4 body when `n_cols % 4 == 0` (4× shorter per-lane chains) plus `simdgroups_per_threadgroup` row indexing so wide rows dispatch 256-thread groups (`norm_batched_geometry`), with a scalar fallback for odd widths. The norm_probe example captures both kernels for future re-profiling. Measurement gotcha: M3 Pro numbers are only trustworthy from isolated criterion groups on a verified-quiet box — three contention incidents on 2026-07-22.

### Fused no-bias layer-norm on all backends
`candle-nn` · `005f01e3` · Incubating · tested (parity vs slow path, cpu/cuda/metal)

The fused path was gated on `bias.is_some()` even though both GPU kernels already branch on a null beta, so `layer_norm_no_bias` never reached a fused kernel at all. `call_layer_norm` beta is now `Option`.

### Norm layers route non-contiguous inputs to the fused kernels
`candle-nn` · `fab85fa9` · Incubating · tested (transposed-input parity vs slow ops, cpu+metal, all three norm forms)

`LayerNorm::forward`/`RmsNorm::forward` now call `.contiguous()` (a no-op clone when already contiguous) instead of falling back to the composed slow path — residual-add views reach the fused kernels. Behaviour note: non-contiguous inputs now take the no-bwd fused path, consistent with existing contiguous behaviour. Removes the need for downstream `.contiguous()` wrappers around every norm call. The accompanying downstream diagnosis that the slow path "materializes f64 means" was refuted — it is f32-internal.

### cumsum: CustomOp CPU/CUDA and Metal scan kernels
`core` / `metal-kernels` · `199e57b6` (carried from open #3700, authorship preserved) + `03e58a1d` · Incubating · tested

Covers a 102720-length case that previously needed a ~42 GB triu allocation, plus layout adaptation, u32/i64, and autograd. The Metal arm is a two-level `simd_shuffle_up` scan, one threadgroup per row, tile loop with carry; f32/u32/i64 (halves excluded, as on CUDA). #3700 is still open upstream — if it lands modified, reconcile at the next merge.

### `grid_sample` and composed `ms_deform_attn`
`core` (all backends) / `candle-nn` · `43816d18` · Incubating · tested

Validated against torch-2.10 fixtures for both `align_corners` modes and the canonical Deformable-DETR `ms_deform_attn_core_pytorch` reference; cpu+metal locally, CUDA on the CUDA box. M3 Pro at an RT-DETR-family MS-DA shape (4 feature levels 64×64→8×8, 8 heads): 39.1 ms on device vs 51.8 ms host round-trip (1.3×), and the result stays on-device — no mid-graph sync, which is the structural win.

`grid_sample` (bilinear, zeros padding) runs one thread per output position looping channels on both GPU backends, with kernels in conv.metal / conv.cu (no new Source plumbing); the rayon CPU implementation is the oracle. `ms_deform_attn` is **composed** from per-level `grid_sample` plus a weighted reduce, and the 39 ms says the composition carries real overhead — a fused MS-DA kernel or a composition profiling pass is the follow-up if layout perf demands it. One hypothesis is already refuted (`4012f021`, reverted in `56896ed5`): replacing the levels stack with per-level weighted accumulation measured **17.7% slower** (46.0 ms), because the extra strided-view muls/sums/adds cost more than one big stack+reduce. The 39 ms is therefore *not* the stack — attribute it with the gpudebug encoder timeline before the next attempt.

### `truncate_to` on RotatingCache and RotatingKvCache
`candle-nn` · `587590d7` + cursor-rewind fix · Incubating · tested

The offset now rewinds relative to the previous cursor; the old `new_len % max` mapping broke after a bulk `seq_len >= max_seq_len` append. Regression-tested in candle-nn/tests/kv_cache.rs. Useful for speculative decoding — dropping rejected drafts' KV. See [Upstream coordination](#upstream-coordination) re #3188.

### Completion hook and kernel timestamps for programmatic profiling
`metal-profile` · `fe62c4e3`, `22839282`, re-grafted in `23b9460e` · Incubating · tested

Completion hook plus `addCompletedHandler`/kernel timestamps. Re-ported onto the post-#3511 fence architecture at the 0.11 merge: the hook installs in `commit_swap_locked` before commit, and `command_encoder_with_buffer` is re-expressed against `CommandsGuard`. Foundation for the profiling-framework branch, which predates the merge and needs the same re-port.

### `MetalDevice::stop_capture`
`metal-core` · `a06fbd2d` · Incubating · tested (the metal_basics example seals the trace)

Pairs with the existing `capture()`; without it, apps capturing gputraces programmatically need their own objc2 call to seal the file. Was a `Wanted` entry.

### Runtime-tunable `compute_per_buffer`
`metal-core` · `169e7e80`, re-grafted in `23b9460e` · Incubating · experimental

Now an `AtomicUsize` on the fence-based `Commands`, plus a buffer `label()` reader. A commit-cadence probe on the downstream streaming-TTS workload (GPU busy 97.1%) says the default is fine there, but the knob is still useful for other workloads. See [Upstream coordination](#upstream-coordination) re #3467 — the lazy backend is the structural version of this.

### Metal device-identity, matmul-layout, and scalar-dtype docs
`candle-core` docs · `20afdb82` · Incubating · n/a (doc comments)

`Device::new_metal`/`MetalDevice` document per-handle identity — queue, fence, and pool state are per handle, so sharing one handle per GPU is a requirement rather than a workaround (ordinal-keyed dedup is noted as an upstream design conversation). `Tensor::matmul` documents the CUDA/Metal layout requirements and the `.contiguous()` remedy. `Tensor::full` warns that bare float literals are f64.

## Incubating on side branches

### Comprehensive Metal profiling framework
`metal-profile` · `feat/metal-profile-comprehensive` (+13 commits, ~4.6k lines) · Incubating · tested

Per-dispatch GPU timestamps, counter sets, os_signpost, chrome-trace, plus docs and CI. Would replace the external gputrace-capture workflow downstream apps use today; propose upstream as a feature-gated module. Predates the 0.11 merge and needs the same post-#3511 re-port the hook layer got in `23b9460e`.

### GatedDeltaNet Metal kernels
`metal-kernels` / `models` · side branch @ `cd2499cc` · Incubating · proven (downstream LLM runner)

Streaming prefill, chunked with GQA, and decode kernels for hybrid linear-attention (gated delta-net) models. Large; upstream appetite unknown — propose as candle-nn ops or examples.

### Ternary/2-bit quantized Metal GEMM family
`metal-quantized` · side branch @ `cd2499cc` · Incubating · proven (downstream runner) / experimental (the GEMV spike)

`mm2d_q2_0` plus split-K, and a bit-plane popcount GEMV spike, for ternary (~2.125 bpw Q2_0) weights. Needs Metal-4 toolchain notes; receipts live in the consuming repo's ticket history. The supporting candle-core/quantized infra changes ride the same branch and need untangling from workload-specific code before any PR.

## Merged upstream

Newest first. #3760 Metal pipeline-cache source identity (merged 2026-07-23) · #3542 Metal debug labels (the `metal-debug-labels` feature downstream apps build on) · #3493 quantized `Cow::Owned` UAF · #3481 Rust 1.95 clippy · #3479 SDPA q_seq>1 routes to full kernel · #3478 copy2d I16/I32 · #3477 RMSNorm f32 overflow · #2086 quantized llama3 example · #2056 Metal unary tiling + benches · #2048 qmatmul benches · #2012 sign op · #2010 Metal dtype extension (unary/binary/reduce) · #2002 Metal reduce refactor · #1995 candle-nn benches · #1986 pub exports · #1938 Metal conv dtypes · #1909 strided index-select · #1903 conv_transpose2d Metal · #1874 conv_transpose1d Metal · #1869 avg_pool2d · #1863 max_pool2d · #1862 index-add dtypes · #1860 Metal cast coverage · #1849 scatter-add f16/bf16.

## Closed with receipts

Kept so nobody retries them blind.

### ug-JIT elementwise-chain fusion
Refuted · receipt: M3 Pro spike (`feat/ug-fusion-spike`, ug_fusion_spike example)

candle-ug 0.5 via `UgIOp1` with default lowering: a fused cos-chain measured 55.4 ms against 27.7 µs composed at 356k f32 — roughly 2000× slower, as if default lowering serializes the whole tensor per thread — and max|diff| was 1.6e-1 against candle's own cos. Found and fixed along the way: the ug+metal feature combination did not compile at 0.11 (`d2acdcd0`), which is worth upstreaming on its own. Caveat: only `lower_op::Opts::default()` was tried, so revisit if ug's Metal codegen matures. Until then, activation and elementwise fusion here is steel-epilogue-shaped or hand-written MSL.

### SDPA BQ=8 full-kernel tiles for q_seq 2–8
Rejected (#3480) + Refuted · receipt: 30× slowdown measured on M3 (revert `336b5800`)

Do not resubmit without a new mechanism.

### `mm2d_q2_0` index-hoist
Refuted · receipt: fork commit `cd2499cc`

Byte-exact and +0.08% tok/s — the address-generation limiter was unmoved.

### Thread-local private-buffers pool
Superseded · `b9376840`, `2300b05e`

Upstream's #3532 fence architecture orders buffer reuse on the GPU. The transplanted claim-then-`wait_until_completed` also cleared `prev_ce_outputs` racily under concurrency, caught by upstream's own `metal_concurrent_tests` at the merge. Upstream allocators restored in `23b9460e`.

### `drop_unused_buffers` sweeps the private pool
Superseded · `fix/metal-private-buffer-pool-sweep` (`cdabfbcf`)

Upstream 0.11 sweeps both pools, with residency-set removal.

### Encoder-label propagation fixes
Superseded · `6f1a3aad`, `f22b3e69`, `8eb56a2d`, `6e2ea560`

Label-after-dispatch clobber, op-level gaps, and gemm/indexing/reduce/cast labels. A prototype predating #3542 (branch point 2026-04-16, PR merged 2026-06-18). The merged debug-group design covers per-dispatch attribution, and per-dispatch `set_label` is wrong on the shared concurrent encoder — dropped at the 0.11 merge.

### 2024-era closed PRs
where_cond Metal perf (#1876), bytemuck refactor (#2053), Metal CI (#2095), Metal random seed (#1959) — historical; see the PRs.

## Wanted

Currently empty. The last two entries — a device-resident top-k, and `grid_sample`/deformable attention — were closed by their implementations on 2026-07-22.

## Open investigations

- `metal_concurrent_tests::concurrent_readback` SIGSEGV'd once under a full parallel suite run (348 tests, many concurrent Metal devices) at `03e58a1d`; 8 isolated runs and 2 subsequent full runs were clean. Likely a rare race in upstream's post-#3511 concurrency machinery — the same family its own tests target. Track frequency; investigate if it recurs.

## Branch hygiene

- Cleaned 2026-07-22: deleted 13 local branches whose content is Merged (#3477/#3478/#3479/#3481/#3493, each verified against upstream before deleting), Rejected with receipts (BQ=8 tiles), or Superseded (label prototype and backups, private-pool sweep, pre-0.11 qmv branch). Remote copies remain on `tomsanbear`. Worktrees pruned.
- Remaining locals: the in-flight PR branches (#3756–#3761), `fix/metal-qmv-tail` (ready to submit), `feat/metal-profile-comprehensive` (needs the post-#3511 re-port), the quantized/linear-attention side branch (tip `cd2499cc` — it carries its consuming project's name, kept because that repo's workflow docs push to it by name; renaming needs coordinating there first), `pr-3700` (reference), `main`, `tomsanbear-dev`.
- In-flight PR branches are behind their `tomsanbear` remotes (review pushed from elsewhere); `git fetch tomsanbear` and fast-forward before touching. `perf/metal-cache-read-locks` is the only one still open — the #3760 branch can be deleted now that it has merged.
