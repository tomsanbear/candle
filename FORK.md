# tomsanbear/candle — fork bill of materials

This fork is the **incubation trunk** for candle changes born in our projects
(wolfrpsiw, lmbrrr, …): fixes and features land on `tomsanbear-dev`, prove
themselves in production use, then graduate to focused `fix/*` / `feat/*` /
`perf/*` topic branches and upstream PRs. This file is the single inventory —
every carried change, its upstream status, and our confidence in it.

**Base**: `tomsanbear-dev` is merged with upstream `main` at 0.11.0
(`31f35b14`, merge commit `23b9460e`, 2026-07-22). Validation at the merge:
candle-metal-kernels 60/60, candle-core `--features metal` 334/334,
candle-nn `--features metal` 75/75.

**Status vocabulary** (adapted from Yocto's `Upstream-Status`, the closest
industry standard for carried patches): `Incubating` (on tomsanbear-dev, not
yet PR'd) · `Submitted (#PR)` (open upstream) · `Merged (#PR)` (in upstream
main) · `Rejected (#PR)` (upstream said no) · `Refuted` (we measured it and
killed it ourselves — kept for the record) · `Superseded` (upstream solved
the same problem another way — kept for the record) · `Wanted` (identified
need, nothing written yet).

**Confidence**: `proven` (shipping in a project with receipts) · `tested`
(unit/bench coverage, not in production) · `experimental` · `n/a`.

**Maintenance protocol** (humans and agents): add a row when a change lands
on `tomsanbear-dev`; move it when a PR opens/merges/closes; never delete
Rejected/Refuted rows — their receipts prevent re-litigating. Keep one row
per logical change, not per commit.

## In flight (open upstream PRs)

All five 2026 PRs below are also carried on `tomsanbear-dev` (cherry-picked
2026-07-22, re-validated on the 0.11 base).

| Item | Area | Branch | Status | Confidence | Notes |
|---|---|---|---|---|---|
| Shared locks for Metal pipeline-cache hits | metal-core | perf/metal-cache-read-locks | Submitted (#3761) | tested | read-mostly cache; contention fix |
| Metal pipeline cache source identity | metal-core | fix/metal-pipeline-cache-identity | Submitted (#3760) | tested | cache key collisions across kernel sources |
| CPU batched matmul with broadcast (stride-0) views | cpu | fix/cpu-stride-zero-batched-matmul | Submitted (#3758) | tested | |
| Empty binary-op validation + gradients | core/autograd | fix/empty-binary-validation-autograd | Submitted (#3757) | tested | |
| Zero-sized matmul validation + gradients | core/autograd | fix/zero-matmul-validation-autograd | Submitted (#3756) | tested | dev carries the 3-arg helper as `assert_zero_grad_shaped` (renamed at cherry-pick to coexist with #3757's 2-arg one) |
| Metal kernels take encoder not command buffer | metal-core | (legacy) | Submitted (#2061) | n/a | 2024-era; likely stale — triage: close or refresh |
| Command encoder/buffer reuse refactor | metal-core | (legacy) | Submitted (#2037) | n/a | 2024-era; superseded by later upstream work — triage |

## Incubating on `tomsanbear-dev` (not yet PR'd)

| Item | Area | Commits | Status | Confidence | Notes |
|---|---|---|---|---|---|
| Metal quantized matvec tail out-of-bounds + dispatch geometry | metal-quantized | d2318473 | Incubating | tested (`quantized_matmul_mv_*_metal` on 0.11) | **no upstream PR yet — needs one** |
| `MetalDevice::stop_capture` | metal-core | phase-0 | Incubating | tested (metal_basics example seals the trace) | pairs with `capture()`; wolfrpsiw can drop its direct objc2 call; was a `Wanted` row |
| conv im2col paths discarded the contiguous kernel copy | core (all 3 backends) | phase-0 | Incubating | tested (regression across cpu/metal/cuda) | pre-existing upstream bug: non-contiguous conv kernels read garbage; good first upstream PR |
| Metal cast i32/i16 coverage (full 8×8 matrix) + generic `to_dtype` device test | metal-kernels | phase-1 | Incubating | tested (kernel-level + first generic to_dtype sweep in the suite) | f64 deliberately excluded (MSL spec 2.1: no double at any feature level); CUDA still lacks i16/i32 casts (skipped in the sweep); wolfrpsiw's t3_llama.rs CPU-round-trip workaround becomes deletable |
| CUDA i64↔f16/bf16 cast kernels | cuda-kernels | phase-1 | Incubating | tested (generic to_dtype sweep on balthasar caught + verifies it) | the sweep found CUDA_ERROR_NOT_FOUND for cast_bf16_i64 — token-ids ↔ half-precision class had no kernels |
| conv_transpose1d padded col2im shim (tensor-level centre-crop rewrite) | core (all 3 backends) | phase-1 | Incubating | **proven: m3 bench 41–191×** (u0 512ch k16s8: 78.9 ms → 441 µs; u1: 68.8 ms → 688 µs; u2: 35.3 ms → 857 µs; f32 and bf16 alike) | unblocks the existing fast path gated on padding==0; PyTorch-referenced padded/output-padded/groups tests added (zero padded coverage existed); supersedes wolfrpsiw's model-side GemmConvTranspose1d once wolfrpsiw points here; prime upstream-PR candidate |
| cumsum: CustomOp CPU/CUDA (carried from open upstream #3700, authorship preserved) + Metal scan kernels | core/metal-kernels | 199e57b6 + metal arm | Incubating | tested (incl. 102720-length case that needed a ~42 GB triu alloc before; layout adaptation; u32/i64; autograd) | Metal arm: two-level simd_shuffle_up scan, one TG per row, tile loop with carry; f32/u32/i64 (halfs excluded like CUDA); if #3700 lands modified upstream, reconcile at next merge |
| Fused no-bias layer-norm on all backends | candle-nn | phase-3 | Incubating | tested (parity vs slow path, cpu/cuda/metal) | the gate was `bias.is_some()` even though both GPU kernels already branch on a null beta — `layer_norm_no_bias` never reached a fused kernel; `call_layer_norm` beta is now `Option` |
| Batched simdgroup-per-row layernorm/rmsnorm + rows>=32 heuristic | metal-kernels | phase-3 | **proven: m3 bench** — layer_norm 1024×512: 2.5–2.7× (43.9→17.6 µs f32); rms 1024×512: 1.5–1.6×; 1024×1024: 1.3–2.3×; rows=1 flat (heuristic keeps stock kernels) | ported from wolfrpsiw metal_fused.rs. **Known issue**: rms f32 at wide rows (1024 cols) measures −11% vs stock — accepted for now; to be fixed IN KERNEL (e.g. vectorized/multi-simdgroup loads for wide f32 rows), not by dispatch gating. Upstream posture: #3150 territory — take the receipts to an issue before PRing |
| Fused GEMM/GEMV bias epilogue + `candle_nn::ops::matmul_bias` | metal-kernels/candle-nn | phase-4 | **proven: m3 bench** — prefill 355×512×512: 2.0× (139.9→69.7 µs f32, 158.6→79.3 bf16); decode 1×1024×4096: ~1–3% (weight-bandwidth bound) | pure Rust wiring — the MLX kernels already ship `_axpby1` gemv variants and the steel `use_out_source`/`do_axpby` epilogue; gotcha: buffer-7 batch strides grow a third C segment when use_out_source is set. `Linear::forward` unchanged (explicit opt-in API); the activation epilogue landed as the follow-up row below |
| Fused GEMM/GEMV **activation** epilogue (relu/gelu/silu) + `candle_nn::ops::matmul_bias_act` | metal-kernels/candle-nn | phase-4 follow-up | Incubating | tested (fused == composed matmul→bias→unary at 1e-5 f32 on-device; per-dtype tolerances for f16/bf16; all steel store branches + both gemv kernels, with and without bias) | function constant 120 (`ushort` activation kind) in mlx_gemm.metal + gemv.metal; formulas mirror unary.metal's urelu/ugelu/usilu and evaluate in the output type, so fused matches the composed chain's precision domain; act==0 folds to identity at pipeline specialization (pipelines cached per constants — gemv now always passes constants). cmk layer supports activation without bias too; candle-nn exposes the bias+act pair — a no-bias `matmul_act` surface is a trivial follow-up if a model needs it. Closes the last `Wanted` row |
| Runtime-tunable `compute_per_buffer` (+ buffer `label()` reader) | metal-core | 169e7e80, re-grafted in 23b9460e | Incubating | experimental | now an `AtomicUsize` on the fence-based `Commands`; wolfrpsiw's cadence probe (GPU busy 97.1%) says default is fine there; knob still useful for other workloads |
| `truncate_to` on RotatingCache/RotatingKvCache | candle-nn | 587590d7 + cursor-rewind fix | Incubating | tested | offset now rewinds relative to the previous cursor (the old `new_len % max` mapping broke after a bulk `seq_len >= max_seq_len` append); regression-tested in candle-nn/tests/kv_cache.rs; wolfrpsiw has its own truncate pattern — converge when adopting KvCache |
| Completion hook + `addCompletedHandler`/kernel timestamps for programmatic profiling | metal-profile | fe62c4e3, 22839282, re-grafted in 23b9460e | Incubating | tested | re-ported onto the post-#3511 fence architecture at the 0.11 merge: hook installs in `commit_swap_locked` before commit; `command_encoder_with_buffer` re-expressed against `CommandsGuard`. Foundation for the profiling framework branch, which predates the merge and needs the same re-port |
| Thread-local private-buffers pool (concurrent reuse race) | metal-core | b9376840, 2300b05e | Superseded | n/a | upstream's #3532 fence architecture orders buffer reuse on the GPU; the transplanted claim-then-`wait_until_completed` also cleared `prev_ce_outputs` racily under concurrency (caught by upstream's `metal_concurrent_tests` at the merge). Upstream allocators restored in 23b9460e |
| `drop_unused_buffers` sweeps the private pool | metal-core | fix/metal-private-buffer-pool-sweep (cdabfbcf) | Superseded | n/a | upstream 0.11 sweeps both pools (with residency-set removal) |
| Encoder-label propagation fixes (label-after-dispatch clobber; op-level gaps; gemm/indexing/reduce/cast labels) | metal-debug | 6f1a3aad, f22b3e69, 8eb56a2d, 6e2ea560 | Superseded | n/a | prototype predating #3542's merge (branch point 2026-04-16, PR merged 2026-06-18); the merged debug-group design covers per-dispatch attribution, and per-dispatch `set_label` is wrong on the shared concurrent encoder — dropped at the 0.11 merge |

## Incubating on side branches

| Item | Area | Branch | Status | Confidence | Notes |
|---|---|---|---|---|---|
| Comprehensive Metal profiling framework (per-dispatch GPU timestamps, counter sets, os_signpost, chrome-trace, docs/CI) | metal-profile | feat/metal-profile-comprehensive (+13, ~4.6k lines) | Incubating | tested | would have replaced most of wolfrpsiw's gpudebug workflow; propose upstream as a feature-gated module. Predates the 0.11 merge — needs the same post-#3511 re-port the hook layer got in 23b9460e |
| GatedDeltaNet Metal kernels (streaming prefill, chunked w/ GQA, decode) | metal-kernels/models | lmbrrr | Incubating | proven (lmbrrr) | large; upstream appetite unknown — propose as candle-nn ops or examples |
| Ternary/2-bit quantized Metal GEMM family (`mm2d_q2_0` + split-K, bit-plane popcount GEMV spike) | metal-quantized | lmbrrr | Incubating | proven (lmbrrr) / experimental (B3 spike) | needs Metal-4 toolchain notes; see lmbrrr repo receipts |
| Quantized support/infra changes backing the above | quantized | lmbrrr (candle-core/quantized) | Incubating | proven (lmbrrr) | untangle from campaign-specific code before PR |

## Merged upstream (this fork's landed history, newest first)

#3542 Metal debug labels (the `metal-debug-labels` feature wolfrpsiw builds on) ·
#3493 quantized `Cow::Owned` UAF · #3481 Rust 1.95 clippy · #3479 SDPA q_seq>1
routes to full kernel · #3478 copy2d I16/I32 · #3477 RMSNorm f32 overflow ·
#2086 quantized llama3 example · #2056 Metal unary tiling + benches · #2048
qmatmul benches · #2012 sign op · #2010 Metal dtype extension (unary/binary/
reduce) · #2002 Metal reduce refactor · #1995 candle-nn benches · #1986 pub
exports · #1938 Metal conv dtypes · #1909 strided index-select · #1903
conv_transpose2d Metal · #1874 conv_transpose1d Metal · #1869 avg_pool2d ·
#1863 max_pool2d · #1862 index-add dtypes · #1860 Metal cast coverage · #1849
scatter-add f16/bf16.

## Rejected / refuted (kept so nobody retries them blind)

| Item | Status | Receipt |
|---|---|---|
| ug-JIT elementwise-chain fusion (candle-ug 0.5 via `UgIOp1`, default lowering) | Refuted | m3 spike (`feat/ug-fusion-spike`, ug_fusion_spike example): fused cos-chain 55.4 ms vs 27.7 µs composed at 356k f32 (~2000× slower — default lowering appears to serialize the whole tensor per thread), and max\|diff\| 1.6e-1 vs candle's cos. Also found+fixed: the ug+metal feature combo didn't compile at 0.11 (d2acdcd0). Caveat: only `lower_op::Opts::default()` was tried; revisit if ug's Metal codegen matures. Until then activation/elementwise fusion is steel-epilogue-shaped or hand-written MSL |
| SDPA BQ=8 full-kernel tiles for q_seq 2–8 | Rejected (#3480) + Refuted | measured 30× slowdown on M3 (revert 336b5800); do not resubmit without a new mechanism |
| mm2d_q2_0 index-hoist | Refuted | byte-exact, +0.08% tok/s — address-gen limiter unmoved (lmbrrr cd2499cc) |
| where_cond Metal perf (#1876), bytemuck refactor (#2053), Metal CI (#2095), Metal random seed (#1959) | Closed 2024 | historical; see PRs |

## Gap analysis — `Wanted`: needs identified by projects, nothing written yet

| Item | Area | Motivating project + receipt | Suggested shape |
|---|---|---|---|

*(empty — the last row, the GEMM activation epilogue, landed on dev 2026-07-22; add new rows as project campaigns surface needs)*

## Open investigations

- `metal_concurrent_tests::concurrent_readback` SIGSEGV'd once under a full
  parallel suite run (348 tests, many concurrent Metal devices) at 03e58a1d;
  8 isolated runs + 2 subsequent full runs were clean. Likely a rare race in
  upstream's post-#3511 concurrency machinery (the same family its own tests
  target). Track frequency; investigate if it recurs.

## Branch hygiene notes

- Cleaned 2026-07-22: deleted 13 local branches whose content is Merged
  (#3477/#3478/#3479/#3481/#3493 — verified against upstream before
  deleting), Rejected with receipts (BQ=8 tiles), or Superseded (label
  prototype + backups, private-pool sweep, pre-0.11 qmv branch). Remote
  copies remain on `tomsanbear`. Worktrees pruned.
- Remaining locals: the 5 in-flight PR branches (#3756–#3761),
  `fix/metal-qmv-tail` (ready to submit), `feat/metal-profile-comprehensive`
  (needs the post-#3511 re-port), `lmbrrr`, `pr-3700` (reference), `main`,
  `tomsanbear-dev`.
- In-flight PR branches are behind their `tomsanbear` remotes (review pushed
  from elsewhere); `git fetch tomsanbear` + fast-forward before touching.
