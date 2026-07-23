# tomsanbear/candle — fork bill of materials

This fork is the **incubation trunk** for candle changes born in downstream production use — an on-device streaming TTS app (token LM → flow decoder → HiFi-GAN-style vocoder on Metal), a PDF→Markdown document-conversion pipeline (RT-DETR-family layout detection, OCR, table structure), and a local-LLM inference runner for ternary-quantized hybrid linear-attention models on Apple Silicon. Fixes and features land on `tomsanbear-dev`, prove themselves in production use, then graduate to focused `fix/*` / `feat/*` / `perf/*` topic branches and upstream PRs. This file is the single inventory — every carried change, its upstream status, and our confidence in it.

**Base**: `tomsanbear-dev` is merged with upstream `main` at 0.11.0 (`31f35b14`, merge commit `23b9460e`, 2026-07-22). Validation at the merge: candle-metal-kernels 60/60, candle-core `--features metal` 334/334, candle-nn `--features metal` 75/75.

**Status vocabulary** (adapted from Yocto's `Upstream-Status`, the closest industry standard for carried patches): `Incubating` (on tomsanbear-dev, not yet PR'd) · `Submitted (#PR)` (open upstream) · `Merged (#PR)` (in upstream main) · `Rejected (#PR)` (upstream said no) · `Refuted` (we measured it and killed it ourselves — kept for the record) · `Superseded` (upstream solved the same problem another way — kept for the record) · `Wanted` (identified need, nothing written yet).

**Confidence**: `proven` (shipping in a downstream app with receipts) · `tested` (unit/bench coverage, not in production) · `experimental` · `n/a`.

**Maintenance protocol** (humans and agents): add an entry when a change lands on `tomsanbear-dev`; move it when a PR opens/merges/closes; never delete Rejected/Refuted entries — their receipts prevent re-litigating. Keep one entry per logical change, not per commit, and keep the Contents index in sync.

## Contents

- **[In flight (open upstream PRs)](#in-flight-open-upstream-prs)** — submitted or carried changes awaiting upstream action
  - [Shared locks for Metal pipeline-cache hits](#shared-locks-for-metal-pipeline-cache-hits) — read-mostly cache contention fix (#3761)
  - [Metal pipeline-cache source identity](#metal-pipeline-cache-source-identity) — cache-key collisions across kernel sources (#3760)
  - [CPU batched matmul with broadcast (stride-0) views](#cpu-batched-matmul-with-broadcast-stride-0-views) — (#3758)
  - [Empty binary-op validation and gradients](#empty-binary-op-validation-and-gradients) — (#3757)
  - [Zero-sized matmul validation and gradients](#zero-sized-matmul-validation-and-gradients) — (#3756)
  - [Metal kernels take an encoder, not a command buffer (#2061)](#metal-kernels-take-an-encoder-not-a-command-buffer-2061) — 2024-era, triage
  - [Command encoder/buffer reuse refactor (#2037)](#command-encoderbuffer-reuse-refactor-2037) — 2024-era, triage
  - [Rust 1.97 clippy fixes and LogitsProcessor NaN guard (carried #3754)](#rust-197-clippy-fixes-and-logitsprocessor-nan-guard-carried-3754) — third-party PR cherry-picked onto dev
- **[Incubating on `tomsanbear-dev` (not yet PR'd)](#incubating-on-tomsanbear-dev-not-yet-prd)** — landed on dev, proving out
  - [Metal quantized matvec tail out-of-bounds and dispatch geometry](#metal-quantized-matvec-tail-out-of-bounds-and-dispatch-geometry) — needs a PR
  - [MetalDevice::stop_capture](#metaldevicestop_capture) — seal gputrace files programmatically
  - [conv im2col paths discarded the contiguous kernel copy](#conv-im2col-paths-discarded-the-contiguous-kernel-copy) — pre-existing upstream bug, all 3 backends
  - [Metal i32/i16 cast coverage and a generic to_dtype device test](#metal-i32i16-cast-coverage-and-a-generic-to_dtype-device-test) — kills CPU-round-trip cast workarounds
  - [CUDA i64↔f16/bf16 cast kernels](#cuda-i64f16bf16-cast-kernels) — token-ids ↔ half-precision had no kernels
  - [conv_transpose1d padded col2im fast path](#conv_transpose1d-padded-col2im-fast-path) — proven 41–191× on vocoder shapes
  - [cumsum: CustomOp CPU/CUDA and Metal scan kernels](#cumsum-customop-cpucuda-and-metal-scan-kernels) — carries open #3700 plus a Metal arm
  - [Fused no-bias layer-norm on all backends](#fused-no-bias-layer-norm-on-all-backends) — the fused path was gated on bias presence
  - [Batched simdgroup-per-row layernorm/rmsnorm](#batched-simdgroup-per-row-layernormrmsnorm) — proven 1.3–2.7× on many-small-rows shapes
  - [Wide-row fix for the batched norms](#wide-row-fix-for-the-batched-norms) — vec4 loads + flexible threadgroups, occupancy root-cause
  - [Fused GEMM/GEMV bias epilogue](#fused-gemmgemv-bias-epilogue) — proven 2.0× prefill via ops::matmul_bias
  - [Fused GEMM/GEMV activation epilogue](#fused-gemmgemv-activation-epilogue) — relu/gelu/silu free in the epilogue
  - [Runtime-tunable compute_per_buffer](#runtime-tunable-compute_per_buffer) — commit-cadence knob
  - [truncate_to on RotatingCache and RotatingKvCache](#truncate_to-on-rotatingcache-and-rotatingkvcache) — cursor-rewind fix included
  - [Completion hook and kernel timestamps for programmatic profiling](#completion-hook-and-kernel-timestamps-for-programmatic-profiling) — foundation for the profiling framework branch
  - [Metal f64 casts via software bit-conversion](#metal-f64-casts-via-software-bit-conversion) — full 9×9 to_dtype matrix without MSL double
  - [Norm layers route non-contiguous inputs to the fused kernels](#norm-layers-route-non-contiguous-inputs-to-the-fused-kernels) — residual-add views reach the fast path
  - [Metal device-identity, matmul-layout, and scalar-dtype docs](#metal-device-identity-matmul-layout-and-scalar-dtype-docs) — doc comments only
  - [grid_sample and composed ms_deform_attn](#grid_sample-and-composed-ms_deform_attn) — bilinear/zeros on cpu/cuda/metal, 1.3× over host round-trip
  - [Wide-row top-k kernel and wide arg_sort routing](#wide-row-top-k-kernel-and-wide-arg_sort-routing) — proven 3.0×; also fixes wide arg_sort not running at all
  - [Thread-local private-buffers pool (superseded)](#thread-local-private-buffers-pool-superseded)
  - [drop_unused_buffers sweeps the private pool (superseded)](#drop_unused_buffers-sweeps-the-private-pool-superseded)
  - [Encoder-label propagation fixes (superseded)](#encoder-label-propagation-fixes-superseded)
- **[Incubating on side branches](#incubating-on-side-branches)** — larger work not yet on dev
  - [Comprehensive Metal profiling framework](#comprehensive-metal-profiling-framework) — ~4.6k lines, needs post-#3511 re-port
  - [GatedDeltaNet Metal kernels](#gateddeltanet-metal-kernels) — streaming prefill, chunked w/ GQA, decode
  - [Ternary/2-bit quantized Metal GEMM family](#ternary2-bit-quantized-metal-gemm-family) — mm2d_q2_0 + split-K + GEMV spike
  - [Quantized support/infra changes](#quantized-supportinfra-changes) — backing the ternary work
- **[Merged upstream](#merged-upstream-this-forks-landed-history-newest-first)** — the fork's landed history
- **[Rejected / refuted](#rejected--refuted-kept-so-nobody-retries-them-blind)** — kept so nobody retries them blind
  - [ug-JIT elementwise-chain fusion (refuted)](#ug-jit-elementwise-chain-fusion-refuted) — ~2000× slower than composed at the spike shape
  - [SDPA BQ=8 full-kernel tiles for q_seq 2–8 (rejected, refuted)](#sdpa-bq8-full-kernel-tiles-for-q_seq-28-rejected-refuted) — 30× slowdown on M3
  - [mm2d_q2_0 index-hoist (refuted)](#mm2d_q2_0-index-hoist-refuted) — byte-exact, +0.08% tok/s
  - [2024-era closed PRs](#2024-era-closed-prs)
- **[Wanted](#wanted)** — identified needs with nothing written yet (currently empty)
- **[Open investigations](#open-investigations)** — one-off flakes being tracked
- **[Branch hygiene notes](#branch-hygiene-notes)** — local/remote branch state

## In flight (open upstream PRs)

All five 2026 PRs below are also carried on `tomsanbear-dev` (cherry-picked 2026-07-22, re-validated on the 0.11 base).

### Shared locks for Metal pipeline-cache hits

**Area**: metal-core · **Branch**: `perf/metal-cache-read-locks` · **Status**: Submitted (#3761) · **Confidence**: tested

The pipeline cache is read-mostly; taking the exclusive lock on every hit was a contention point. Hits now take a shared read lock.

### Metal pipeline-cache source identity

**Area**: metal-core · **Branch**: `fix/metal-pipeline-cache-identity` · **Status**: Submitted (#3760) · **Confidence**: tested

The pipeline-cache key could collide across kernel sources; the source identity is now part of the key.

### CPU batched matmul with broadcast (stride-0) views

**Area**: cpu · **Branch**: `fix/cpu-stride-zero-batched-matmul` · **Status**: Submitted (#3758) · **Confidence**: tested

### Empty binary-op validation and gradients

**Area**: core/autograd · **Branch**: `fix/empty-binary-validation-autograd` · **Status**: Submitted (#3757) · **Confidence**: tested

### Zero-sized matmul validation and gradients

**Area**: core/autograd · **Branch**: `fix/zero-matmul-validation-autograd` · **Status**: Submitted (#3756) · **Confidence**: tested

Dev carries the 3-arg helper as `assert_zero_grad_shaped` (renamed at cherry-pick to coexist with #3757's 2-arg one).

### Metal kernels take an encoder, not a command buffer (#2061)

**Area**: metal-core · **Branch**: (legacy) · **Status**: Submitted (#2061) · **Confidence**: n/a

2024-era; likely stale — triage: close or refresh.

### Command encoder/buffer reuse refactor (#2037)

**Area**: metal-core · **Branch**: (legacy) · **Status**: Submitted (#2037) · **Confidence**: n/a

2024-era; superseded by later upstream work — triage.

### Rust 1.97 clippy fixes and LogitsProcessor NaN guard (carried #3754)

**Area**: workspace · **Branch**: third-party #3754 (GregoryBolshakov) · **Status**: Carried · **Confidence**: tested (clippy clean on core/transformers after)

Cherry-picked (`-x`) onto dev 2026-07-22, authorship preserved; also clears the long-standing shape.rs warning. Drops out naturally at the next upstream merge once #3754 lands.

## Incubating on `tomsanbear-dev` (not yet PR'd)

### Metal quantized matvec tail out-of-bounds and dispatch geometry

**Area**: metal-quantized · **Commit**: `d2318473` · **Status**: Incubating · **Confidence**: tested (`quantized_matmul_mv_*_metal` on 0.11)

**No upstream PR yet — needs one.**

### MetalDevice::stop_capture

**Area**: metal-core · **Commit**: `a06fbd2d` · **Status**: Incubating · **Confidence**: tested (metal_basics example seals the trace)

Pairs with `capture()`. Downstream apps that capture gputraces programmatically can drop their direct objc2 calls. Was a `Wanted` entry.

### conv im2col paths discarded the contiguous kernel copy

**Area**: core (all 3 backends) · **Commit**: `14ae55a3` · **Status**: Incubating · **Confidence**: tested (regression across cpu/metal/cuda)

Pre-existing upstream bug: non-contiguous conv kernels read garbage. Good first upstream PR.

### Metal i32/i16 cast coverage and a generic to_dtype device test

**Area**: metal-kernels · **Commit**: `6b02745d` · **Status**: Incubating · **Confidence**: tested (kernel-level + first generic to_dtype sweep in the suite)

Fills the i32/i16 rows and columns of the Metal cast matrix and adds the suite's first generic `to_dtype` device sweep. f64 was deliberately excluded here (MSL spec 2.1: no double at any feature level) and landed later via the software bit-conversion entry below; CUDA still lacks i16/i32 casts (skipped in the sweep). Kills the class of downstream workaround where token-id tensors are moved to the CPU purely to widen i32→i64 for an embedding `index_select`, then moved back.

### CUDA i64↔f16/bf16 cast kernels

**Area**: cuda-kernels · **Commit**: `13adaa2a` · **Status**: Incubating · **Confidence**: tested (generic to_dtype sweep on the CUDA test box caught + verifies it)

The sweep found `CUDA_ERROR_NOT_FOUND` for `cast_bf16_i64` — the token-ids ↔ half-precision class had no kernels at all.

### conv_transpose1d padded col2im fast path

**Area**: core (all 3 backends) · **Commit**: `70452db0` · **Status**: Incubating · **Confidence**: **proven — M3 Pro bench, 41–191×** (upsampler stages of a HiFi-GAN-style vocoder, f32 and bf16 alike: 512ch k16s8 78.9 ms → 441 µs; 68.8 ms → 688 µs; 35.3 ms → 857 µs)

A tensor-level centre-crop rewrite of the padded col2im shim unblocks the existing fast path that was gated on `padding == 0`. PyTorch-referenced padded/output-padded/groups tests added (zero padded coverage existed). Supersedes the model-side GEMM-based conv-transpose workaround the downstream TTS vocoder carried, once that app repins. Prime upstream-PR candidate.

### cumsum: CustomOp CPU/CUDA and Metal scan kernels

**Area**: core/metal-kernels · **Commits**: `199e57b6` (carried from open upstream #3700, authorship preserved) + `03e58a1d` (Metal arm) · **Status**: Incubating · **Confidence**: tested (incl. a 102720-length case that needed a ~42 GB triu alloc before; layout adaptation; u32/i64; autograd)

Metal arm: two-level simd_shuffle_up scan, one threadgroup per row, tile loop with carry; f32/u32/i64 (halfs excluded like CUDA). If #3700 lands modified upstream, reconcile at the next merge.

### Fused no-bias layer-norm on all backends

**Area**: candle-nn · **Commit**: `005f01e3` · **Status**: Incubating · **Confidence**: tested (parity vs slow path, cpu/cuda/metal)

The gate was `bias.is_some()` even though both GPU kernels already branch on a null beta — `layer_norm_no_bias` never reached a fused kernel. `call_layer_norm` beta is now `Option`.

### Batched simdgroup-per-row layernorm/rmsnorm

**Area**: metal-kernels · **Commit**: `8e9f701a` · **Status**: Incubating · **Confidence**: **proven — M3 Pro bench**: layer_norm 1024×512 2.5–2.7× (43.9→17.6 µs f32); rms 1024×512 1.5–1.6×; 1024×1024 1.3–2.3×; rows=1 flat (heuristic keeps stock kernels)

Batched kernels with a rows>=32 routing heuristic, ported from the downstream TTS app's private fused-norm kernels. The former known issue (rms f32 wide rows −11% vs stock) is RESOLVED in kernel — next entry. Upstream posture: #3150 territory — take the receipts to an issue before PRing.

### Wide-row fix for the batched norms

**Area**: metal-kernels · **Commit**: `e8f52701` · **Status**: Incubating · **Confidence**: **proven — M3 Pro bench, isolated groups**: rms f32 1024×1024 45.2→37.2 µs (−18%, flips −11%-vs-stock to ~+8%); rms f32 1024×512 17.2→15.4 µs; rms bf16 1024×1024 25.7→12.4 µs (−52%); layer_norm f32 1024×1024 46.1→41.1 µs; decode rows flat (stock path untouched, 2.64 µs)

Vec4 loads + flexible rows-per-threadgroup. Root-caused with gpudebug counters (M3 Pro, macOS 27, gpudebug 1.0): 32-row 1024-thread groups reached 45% occupancy vs an 86% manager target with l1 evictions at 0 and launch limiter at 1% — occupancy quantization + latency-exposed scalar load chains, NOT L1 thrash (hypothesis refuted; stock's costs are an 80–85% launch limiter and l1 evictions 25). Fix: vec4 body when `n_cols % 4 == 0` (4× shorter per-lane chains) + `simdgroups_per_threadgroup` row indexing so wide rows dispatch 256-thread groups (`norm_batched_geometry`); scalar fallback for odd widths. The norm_probe example captures both kernels for future re-profiling. Measurement gotcha: M3 Pro numbers are only trustworthy from isolated criterion groups on a verified-quiet box (three contention incidents 2026-07-22).

### Fused GEMM/GEMV bias epilogue

**Area**: metal-kernels/candle-nn · **Commits**: `2ee72f8e` + `0b182601` (`candle_nn::ops::matmul_bias`) · **Status**: Incubating · **Confidence**: **proven — M3 Pro bench**: prefill 355×512×512 2.0× (139.9→69.7 µs f32, 158.6→79.3 bf16); decode 1×1024×4096 ~1–3% (weight-bandwidth bound)

Pure Rust wiring — the MLX kernels already ship `_axpby1` gemv variants and the steel `use_out_source`/`do_axpby` epilogue. Gotcha: buffer-7 batch strides grow a third C segment when `use_out_source` is set. `Linear::forward` unchanged (explicit opt-in API); the activation epilogue landed as the follow-up entry below.

### Fused GEMM/GEMV activation epilogue

**Area**: metal-kernels/candle-nn · **Commit**: `d37a2fa5` (`candle_nn::ops::matmul_bias_act`, relu/gelu/silu) · **Status**: Incubating · **Confidence**: **proven — M3 Pro bench**: prefill 355×512×512 silu fused 69.7 µs ≈ plain matmul_bias 69.9 µs vs 73.0 µs with a separate silu dispatch (f32; bf16 84.4→81.7) — the activation is free in the epilogue; decode 1×1024×4096 weight-bandwidth-bound (~1–2%)

Function constant 120 (`ushort` activation kind) in mlx_gemm.metal + gemv.metal; formulas mirror unary.metal's urelu/ugelu/usilu and evaluate in the output type, so fused matches the composed chain's precision domain (verified at 1e-5 on-device across all steel store branches + both gemv kernels, with and without bias); act==0 folds to identity at pipeline specialization. The candle-metal-kernels layer supports activation without bias too; a no-bias `matmul_act` nn surface is a trivial follow-up. Bench note: full 16-group sweeps showed one thermal artifact (composed f32 prefill 457 µs in-sweep vs 141 µs isolated) — receipts use isolated/stable readings.

### Runtime-tunable compute_per_buffer

**Area**: metal-core · **Commit**: `169e7e80`, re-grafted in `23b9460e` · **Status**: Incubating · **Confidence**: experimental

Now an `AtomicUsize` on the fence-based `Commands`, plus a buffer `label()` reader. A commit-cadence probe in the downstream streaming-TTS workload (GPU busy 97.1%) says the default is fine there; the knob is still useful for other workloads.

### truncate_to on RotatingCache and RotatingKvCache

**Area**: candle-nn · **Commits**: `587590d7` + cursor-rewind fix · **Status**: Incubating · **Confidence**: tested

Offset now rewinds relative to the previous cursor (the old `new_len % max` mapping broke after a bulk `seq_len >= max_seq_len` append); regression-tested in candle-nn/tests/kv_cache.rs. Downstream apps carrying their own truncate patterns can converge on this when adopting KvCache.

### Completion hook and kernel timestamps for programmatic profiling

**Area**: metal-profile · **Commits**: `fe62c4e3`, `22839282`, re-grafted in `23b9460e` · **Status**: Incubating · **Confidence**: tested

Completion hook + `addCompletedHandler`/kernel timestamps. Re-ported onto the post-#3511 fence architecture at the 0.11 merge: the hook installs in `commit_swap_locked` before commit; `command_encoder_with_buffer` re-expressed against `CommandsGuard`. Foundation for the profiling framework branch, which predates the merge and needs the same re-port.

### Metal f64 casts via software bit-conversion

**Area**: metal-kernels/core · **Commit**: `d6a3585a` · **Status**: Incubating · **Confidence**: tested (kernel-level cast_f64 incl. RNE ties/specials/saturation/2^40+ exactness; the generic to_dtype sweep now runs the full 9×9 matrix on Metal, contiguous + strided)

MSL has no double at any feature level (spec §2.1, re-verified incl. feature tables), but casts don't need double *arithmetic*: f64 buffers are read/written as `ulong` and converted in software (RNE f64→f32, exact f32→f64, exact int↔f64 to 2^53 with Rust-`as` saturation). Unblocked a downstream P0: the document-conversion pipeline hit "Metal contiguous to_dtype F64 F16/F32 not implemented" from host-side f64 scalars. The formerly-pending CUDA-side check is resolved: CUDA casts f64 natively and the sweep now includes f64 there too (`7a4c7910`).

### Norm layers route non-contiguous inputs to the fused kernels

**Area**: candle-nn · **Commit**: `fab85fa9` · **Status**: Incubating · **Confidence**: tested (transposed-input parity vs slow ops, cpu+metal, all three norm forms)

`LayerNorm::forward`/`RmsNorm::forward` now `.contiguous()` (a no-op clone when already so) instead of falling back to the composed slow path on non-contiguous input — residual-add views reach the fused kernels. Behaviour note: non-contiguous inputs now take the no-bwd fused path, consistent with the existing contiguous behaviour. Removes the need for downstream `.contiguous()` wrapper helpers around every norm call; the accompanying downstream diagnosis that the norm slow path "materializes f64 means" was refuted (the slow path is f32-internal).

### Metal device-identity, matmul-layout, and scalar-dtype docs

**Area**: candle-core docs · **Commit**: `20afdb82` · **Status**: Incubating · **Confidence**: n/a (doc comments)

`Device::new_metal`/`MetalDevice` document per-handle identity (queue/fence/pool state is per handle — sharing one handle per GPU is required, not a workaround; ordinal-keyed dedup noted as an upstream design conversation); `Tensor::matmul` documents CUDA/Metal layout requirements + the `.contiguous()` remedy; `Tensor::full` warns that bare float literals are f64.

### grid_sample and composed ms_deform_attn

**Area**: all backends + candle-nn · **Commit**: `43816d18` · **Status**: Incubating · **Confidence**: tested (torch-2.10 fixtures for both align_corners modes + the canonical Deformable-DETR `ms_deform_attn_core_pytorch` reference; cpu+metal locally, CUDA on the test box). **M3 Pro bench** at a layout-detection MS-DA shape (RT-DETR family: 4 feature levels 64×64→8×8, 8 heads): device 39.1 ms vs 51.8 ms host round-trip (1.3×) — and the result stays on-device (no mid-graph sync), which is the structural win

`grid_sample` (bilinear, zeros padding) with one thread per output position looping channels on both GPU backends (kernels live in conv.metal / conv.cu — no new Source plumbing); the rayon CPU impl is the oracle. `ms_deform_attn` is COMPOSED from per-level grid_sample + weighted reduce; the 39 ms device time says the composition carries real overhead — a fused MS-DA kernel and/or a composition profiling pass is the follow-up if layout perf demands it. Already refuted (`4012f021`, reverted `56896ed5`): replacing the levels stack with per-level weighted accumulation measured 17.7% SLOWER (46.0 ms) — the extra strided-view muls/sums/adds cost more than the one big stack+reduce, so the 39 ms is NOT the stack; attribute with the gpudebug encoder timeline before the next attempt. Closed the last `Wanted` entry.

### Wide-row top-k kernel and wide arg_sort routing

**Area**: metal-kernels/core/candle-nn · **Commit**: `2c6e5abd` (`candle_nn::ops::topk` + wide `arg_sort` routing) · **Status**: Incubating · **Confidence**: **proven — M3 Pro bench, isolated**: RT-DETR-style query selection (8×24000, k=300) device 1.04 ms vs 3.09 ms host round-trip (3.0×), and the result stays on-device (no mid-graph sync). CUDA: composed fallback verified on the test box (asort_big covers wide rows there)

Discovered en route: Metal `arg_sort_last_dim` beyond 1024 columns didn't run at all (the bitonic kernel dispatches ncols_pad threads in ONE threadgroup) — wide ascending sorts now route to the previously-unwired MLX multi-block kernels; wide descending bails with a clear message. topk kernel: tile-and-merge (bitonic tile sort + log2(2k)-stage bitonic merge with a descending carry), k padded to pow2, f32/f16/bf16. At 8 rows the kernel underfills the GPU (8 threadgroups) — multi-TG-per-row is the optimization if a workload ever needs it.

### Thread-local private-buffers pool (superseded)

**Area**: metal-core · **Commits**: `b9376840`, `2300b05e` · **Status**: Superseded · **Confidence**: n/a

Upstream's #3532 fence architecture orders buffer reuse on the GPU; the transplanted claim-then-`wait_until_completed` also cleared `prev_ce_outputs` racily under concurrency (caught by upstream's `metal_concurrent_tests` at the merge). Upstream allocators restored in `23b9460e`.

### drop_unused_buffers sweeps the private pool (superseded)

**Area**: metal-core · **Branch**: `fix/metal-private-buffer-pool-sweep` (`cdabfbcf`) · **Status**: Superseded · **Confidence**: n/a

Upstream 0.11 sweeps both pools (with residency-set removal).

### Encoder-label propagation fixes (superseded)

**Area**: metal-debug · **Commits**: `6f1a3aad`, `f22b3e69`, `8eb56a2d`, `6e2ea560` · **Status**: Superseded · **Confidence**: n/a

Label-after-dispatch clobber, op-level gaps, gemm/indexing/reduce/cast labels. Prototype predating #3542's merge (branch point 2026-04-16, PR merged 2026-06-18); the merged debug-group design covers per-dispatch attribution, and per-dispatch `set_label` is wrong on the shared concurrent encoder — dropped at the 0.11 merge.

## Incubating on side branches

### Comprehensive Metal profiling framework

**Area**: metal-profile · **Branch**: `feat/metal-profile-comprehensive` (+13 commits, ~4.6k lines) · **Status**: Incubating · **Confidence**: tested

Per-dispatch GPU timestamps, counter sets, os_signpost, chrome-trace, docs/CI. Would replace the external gputrace-capture workflow downstream apps currently use for profiling; propose upstream as a feature-gated module. Predates the 0.11 merge — needs the same post-#3511 re-port the hook layer got in `23b9460e`.

### GatedDeltaNet Metal kernels

**Area**: metal-kernels/models · **Branch**: side branch @ `cd2499cc` · **Status**: Incubating · **Confidence**: proven (shipping in the downstream local-LLM inference runner)

Streaming prefill, chunked with GQA, and decode kernels for hybrid linear-attention (gated delta-net) models. Large; upstream appetite unknown — propose as candle-nn ops or examples.

### Ternary/2-bit quantized Metal GEMM family

**Area**: metal-quantized · **Branch**: side branch @ `cd2499cc` · **Status**: Incubating · **Confidence**: proven (downstream runner) / experimental (the bit-plane popcount GEMV spike)

`mm2d_q2_0` + split-K, plus a bit-plane popcount GEMV spike, for ternary (~2.125 bpw Q2_0) weights. Needs Metal-4 toolchain notes; receipts live in the consuming repo's ticket history.

### Quantized support/infra changes

**Area**: quantized · **Branch**: side branch @ `cd2499cc` (candle-core/quantized) · **Status**: Incubating · **Confidence**: proven (downstream runner)

Support/infra backing the ternary GEMM family. Untangle from workload-specific code before PR.

## Merged upstream (this fork's landed history, newest first)

#3542 Metal debug labels (the `metal-debug-labels` feature downstream apps build on) · #3493 quantized `Cow::Owned` UAF · #3481 Rust 1.95 clippy · #3479 SDPA q_seq>1 routes to full kernel · #3478 copy2d I16/I32 · #3477 RMSNorm f32 overflow · #2086 quantized llama3 example · #2056 Metal unary tiling + benches · #2048 qmatmul benches · #2012 sign op · #2010 Metal dtype extension (unary/binary/reduce) · #2002 Metal reduce refactor · #1995 candle-nn benches · #1986 pub exports · #1938 Metal conv dtypes · #1909 strided index-select · #1903 conv_transpose2d Metal · #1874 conv_transpose1d Metal · #1869 avg_pool2d · #1863 max_pool2d · #1862 index-add dtypes · #1860 Metal cast coverage · #1849 scatter-add f16/bf16.

## Rejected / refuted (kept so nobody retries them blind)

### ug-JIT elementwise-chain fusion (refuted)

**Status**: Refuted · **Receipt**: M3 Pro spike (`feat/ug-fusion-spike`, ug_fusion_spike example)

candle-ug 0.5 via `UgIOp1`, default lowering: fused cos-chain 55.4 ms vs 27.7 µs composed at 356k f32 (~2000× slower — default lowering appears to serialize the whole tensor per thread), and max|diff| 1.6e-1 vs candle's cos. Also found+fixed: the ug+metal feature combo didn't compile at 0.11 (`d2acdcd0`). Caveat: only `lower_op::Opts::default()` was tried; revisit if ug's Metal codegen matures. Until then activation/elementwise fusion is steel-epilogue-shaped or hand-written MSL.

### SDPA BQ=8 full-kernel tiles for q_seq 2–8 (rejected, refuted)

**Status**: Rejected (#3480) + Refuted · **Receipt**: measured 30× slowdown on M3 (revert `336b5800`)

Do not resubmit without a new mechanism.

### mm2d_q2_0 index-hoist (refuted)

**Status**: Refuted · **Receipt**: fork commit `cd2499cc`

Byte-exact, +0.08% tok/s — the address-generation limiter was unmoved.

### 2024-era closed PRs

where_cond Metal perf (#1876), bytemuck refactor (#2053), Metal CI (#2095), Metal random seed (#1959) — closed 2024; historical, see the PRs.

## Wanted

Needs identified by downstream workloads with nothing written yet. Currently empty — the last entries (device-resident top-k; grid_sample/deformable attention) were closed 2026-07-22. When a downstream workload identifies a need before any code exists, add an entry here with: area, the motivating use case and its receipt, and the suggested shape of the change.

## Open investigations

- `metal_concurrent_tests::concurrent_readback` SIGSEGV'd once under a full parallel suite run (348 tests, many concurrent Metal devices) at `03e58a1d`; 8 isolated runs + 2 subsequent full runs were clean. Likely a rare race in upstream's post-#3511 concurrency machinery (the same family its own tests target). Track frequency; investigate if it recurs.

## Branch hygiene notes

- Cleaned 2026-07-22: deleted 13 local branches whose content is Merged (#3477/#3478/#3479/#3481/#3493 — verified against upstream before deleting), Rejected with receipts (BQ=8 tiles), or Superseded (label prototype + backups, private-pool sweep, pre-0.11 qmv branch). Remote copies remain on `tomsanbear`. Worktrees pruned.
- Remaining locals: the 5 in-flight PR branches (#3756–#3761), `fix/metal-qmv-tail` (ready to submit), `feat/metal-profile-comprehensive` (needs the post-#3511 re-port), the quantized/linear-attention side branch (tip `cd2499cc` — it carries its consuming project's name, kept because that repo's workflow docs push to it by name; rename needs coordinating there first), `pr-3700` (reference), `main`, `tomsanbear-dev`.
- In-flight PR branches are behind their `tomsanbear` remotes (review pushed from elsewhere); `git fetch tomsanbear` + fast-forward before touching.
