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
| SDPA BQ=8 full-kernel tiles for q_seq 2–8 | Rejected (#3480) + Refuted | measured 30× slowdown on M3 (revert 336b5800); do not resubmit without a new mechanism |
| mm2d_q2_0 index-hoist | Refuted | byte-exact, +0.08% tok/s — address-gen limiter unmoved (lmbrrr cd2499cc) |
| where_cond Metal perf (#1876), bytemuck refactor (#2053), Metal CI (#2095), Metal random seed (#1959) | Closed 2024 | historical; see PRs |

## Gap analysis — `Wanted`: needs identified by projects, nothing written yet

| Item | Area | Motivating project + receipt | Suggested shape |
|---|---|---|---|
| GEMM epilogue fusion (bias + activation) in mlx_gemm | metal-kernels | wolfrpsiw decode: GPU 97% busy yet ~2× off bandwidth roofline; separate bias/GELU dispatches after every GEMM are the largest fusable class. Unreachable from outside candle | epilogue enum on gemm call sites |
| Batched (simdgroup-per-row) layernorm/rmsnorm variant + row-count dispatch heuristic | metal-kernels | wolfrpsiw: stock kernel 5.34 ns/elem at 355–2840×512; working MSL exists in wolfrpsiw `metal_fused.rs` (1e-4 verified); CAUTION: batched shape regresses rows==1 (measured 0.67→0.77 s) — heuristic mandatory | port wolfrpsiw kernel + `rows>=32` gate |
| Elementwise-chain fusion (investigate `ug` JIT first) | metal-core | wolfrpsiw flow: diffuse glue ~half the stage; candle 0.11 ships a `ug` codegen path — experiment before writing kernels | experiment, then decide |

## Branch hygiene notes

- `feat/metal-i16-i32-copy` is a stale pre-split integration branch (its
  content shipped via #3477–#3479 and tomsanbear-dev) — delete after checking.
- `backup/*` branches are pre-rebase snapshots of the label work — delete
  once #3542 follow-ups land. The label prototype itself is now Superseded
  (see above), so these are candidates for deletion outright.
- `fix/metal-private-buffer-pool-sweep` and the thread-local-pool history are
  Superseded as of the 0.11 merge — deletable after verification.
- Local topic branches are behind their `tomsanbear` remotes (review pushed
  from elsewhere); `git fetch tomsanbear` + fast-forward before touching.
- 7 registered worktrees are prunable (`git worktree prune`).
