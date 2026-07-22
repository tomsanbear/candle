# tomsanbear/candle — fork bill of materials

This fork is the **incubation trunk** for candle changes born in our projects
(wolfrpsiw, lmbrrr, …): fixes and features land on `tomsanbear-dev`, prove
themselves in production use, then graduate to focused `fix/*` / `feat/*` /
`perf/*` topic branches and upstream PRs. This file is the single inventory —
every carried change, its upstream status, and our confidence in it.

**Status vocabulary** (adapted from Yocto's `Upstream-Status`, the closest
industry standard for carried patches): `Incubating` (on tomsanbear-dev, not
yet PR'd) · `Submitted (#PR)` (open upstream) · `Merged (#PR)` (in upstream
main) · `Rejected (#PR)` (upstream said no) · `Refuted` (we measured it and
killed it ourselves — kept for the record) · `Wanted` (identified need,
nothing written yet).

**Confidence**: `proven` (shipping in a project with receipts) · `tested`
(unit/bench coverage, not in production) · `experimental` · `n/a`.

**Maintenance protocol** (humans and agents): add a row when a change lands
on `tomsanbear-dev`; move it when a PR opens/merges/closes; never delete
Rejected/Refuted rows — their receipts prevent re-litigating. Keep one row
per logical change, not per commit.

## In flight (open upstream PRs)

| Item | Area | Branch | Status | Confidence | Notes |
|---|---|---|---|---|---|
| Shared locks for Metal pipeline-cache hits | metal-core | perf/metal-cache-read-locks | Submitted (#3761) | tested | read-mostly cache; contention fix |
| Metal pipeline cache source identity | metal-core | fix/metal-pipeline-cache-identity | Submitted (#3760) | tested | cache key collisions across kernel sources |
| CPU batched matmul with broadcast (stride-0) views | cpu | fix/cpu-stride-zero-batched-matmul | Submitted (#3758) | tested | |
| Empty binary-op validation + gradients | core/autograd | fix/empty-binary-validation-autograd | Submitted (#3757) | tested | |
| Zero-sized matmul validation + gradients | core/autograd | fix/zero-matmul-validation-autograd | Submitted (#3756) | tested | |
| Metal kernels take encoder not command buffer | metal-core | (legacy) | Submitted (#2061) | n/a | 2024-era; likely stale — triage: close or refresh |
| Command encoder/buffer reuse refactor | metal-core | (legacy) | Submitted (#2037) | n/a | 2024-era; superseded by later upstream work — triage |

## Incubating on `tomsanbear-dev` (not yet PR'd)

| Item | Area | Commits | Status | Confidence | Notes |
|---|---|---|---|---|---|
| Thread-local private-buffers pool (concurrent reuse race) | metal-core | b9376840, 2300b05e | Incubating | tested | pairs with the pool-sweep fix below |
| `drop_unused_buffers` sweeps the private pool | metal-core | fix/metal-private-buffer-pool-sweep (cdabfbcf) | Incubating | tested | memory growth under long sessions |
| Runtime-tunable `compute_per_buffer` (+ buffer `label()` reader) | metal-core | 169e7e80 | Incubating | experimental | wolfrpsiw's cadence probe (GPU busy 97.1%) says default is fine there; knob still useful for other workloads |
| `truncate_to` on RotatingCache/RotatingKvCache | candle-nn | 587590d7 | Incubating | tested | wolfrpsiw has its own truncate pattern; converge when adopting KvCache |
| Encoder-label propagation fixes (label-after-dispatch clobber; op-level gaps; gemm/indexing/reduce/cast labels) | metal-debug | 6f1a3aad, f22b3e69, 8eb56a2d, 6e2ea560 | Incubating | proven (wolfrpsiw gputrace campaign consumed labels) | follow-up to merged #3542 |
| Completion hook + `addCompletedHandler`/kernel timestamps for programmatic profiling | metal-profile | fe62c4e3, 22839282 | Incubating | tested | foundation for the profiling framework branch |

## Incubating on side branches

| Item | Area | Branch | Status | Confidence | Notes |
|---|---|---|---|---|---|
| Comprehensive Metal profiling framework (per-dispatch GPU timestamps, counter sets, os_signpost, chrome-trace, docs/CI) | metal-profile | feat/metal-profile-comprehensive (+13, ~4.6k lines) | Incubating | tested | would have replaced most of wolfrpsiw's gpudebug workflow; propose upstream as a feature-gated module |
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
| Linear CPU cumsum + Metal scan kernel | core/metal | wolfrpsiw: CPU cumsum quadratic (16.8 s @ (1,9,102720)); Metal fails "Failed to create Buffer" ~100k; workaround = host scan that forces a mid-pipeline sync. Repro: wolfrpsiw `metal_probe --repro-cumsum` | CPU rewrite + decoupled-lookback scan kernel |
| GEMM epilogue fusion (bias + activation) in mlx_gemm | metal-kernels | wolfrpsiw decode: GPU 97% busy yet ~2× off bandwidth roofline; separate bias/GELU dispatches after every GEMM are the largest fusable class. Unreachable from outside candle | epilogue enum on gemm call sites |
| Batched (simdgroup-per-row) layernorm/rmsnorm variant + row-count dispatch heuristic | metal-kernels | wolfrpsiw: stock kernel 5.34 ns/elem at 355–2840×512; working MSL exists in wolfrpsiw `metal_fused.rs` (1e-4 verified); CAUTION: batched shape regresses rows==1 (measured 0.67→0.77 s) — heuristic mandatory | port wolfrpsiw kernel + `rows>=32` gate |
| Metal I32→I64 cast (audit integer cast pairs) | metal-kernels | wolfrpsiw: "Metal contiguous to_dtype I32 I64 not implemented"; CPU round-trip workaround | few-line cast.metal addition |
| `MetalDevice::stop_capture()` | metal-core | wolfrpsiw calls objc2 directly to seal .gputrace files | 2-line API next to `capture()` |
| conv_transpose1d Metal: GEMM lowering or tiled kernel | metal-kernels | wolfrpsiw: original #1874 kernel is a scalar gather — 36% of the vocoder stage (instr-throughput 85%); model-side GEMM lowering (−32% stage wall) in wolfrpsiw `hift.rs` ready to generalize | successor to #1874 |
| Elementwise-chain fusion (investigate `ug` JIT first) | metal-core | wolfrpsiw flow: diffuse glue ~half the stage; candle 0.11 ships a `ug` codegen path — experiment before writing kernels | experiment, then decide |

## Branch hygiene notes

- `feat/metal-i16-i32-copy` is a stale pre-split integration branch (its
  content shipped via #3477–#3479 and tomsanbear-dev) — delete after checking.
- `backup/*` branches are pre-rebase snapshots of the label work — delete
  once #3542 follow-ups land.
- Local topic branches are behind their `tomsanbear` remotes (review pushed
  from elsewhere); `git fetch tomsanbear` + fast-forward before touching.
- 7 registered worktrees are prunable (`git worktree prune`).
