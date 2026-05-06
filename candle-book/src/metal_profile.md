# Metal GPU Profiling

When optimizing GPU kernels on Apple Silicon, CPU-side tracing (e.g.
[tracing-chrome](tracing.md)) is deaf to the GPU: it only sees when Candle's
host thread *enqueued* an operation, not when the GPU actually executed it.
Per-kernel GPU duration, command-buffer scheduling latency, encoder-level
overhead, buffer allocation cost, and resource bindings are all invisible to
the CPU profiler.

Candle's `metal-profile` cargo feature closes that gap. It attaches
`MTLCounterSampleBuffer` to every compute encoder so the GPU's command
processor writes hardware-clock timestamps at each encoder boundary, and
streams the results — alongside per-command-buffer rollups, CPU encode/dispatch
timings, buffer allocations, and full per-encoder context — to a single Chrome
Trace Event Format JSON file. The trace is queryable programmatically with
`jq`, any JSON parser, or Perfetto's `trace_processor` SQL — no GUI in the
loop.

This page is a quick start. The full design + workflow guide, including the
boundaries vs. Xcode `.gputrace` and Instruments / `xctrace`, lives at
[`candle-metal-kernels/METAL_PROFILING.md`](https://github.com/huggingface/candle/blob/main/candle-metal-kernels/METAL_PROFILING.md).

## Quick start

Build with the feature enabled and run one of the worked examples:

```bash
# Synthetic kernel mix, ~30 events:
cargo run -p candle-core --example metal_profile --release \
  --features metal-profile -- \
  --profile-json /tmp/candle-metal-profile.json

# Self-contained Llama-shaped 6-layer decoder, ~30k events / 32 tokens:
cargo run -p candle-examples --example metal_profile_llm --release \
  --features metal-profile -- \
  --decode-steps 32 \
  --profile-json /tmp/candle-metal-profile-llm.json
```

Inspect with [Perfetto UI](https://ui.perfetto.dev/) (drag-and-drop the JSON)
or `chrome://tracing` (click **Load**), or query it from the command line:

```bash
trace_processor -Q "select name, count(*) n, round(sum(dur)/1000.0, 3) total_us
                    from slice where category = 'metal-gpu-encoder'
                    group by name order by sum(dur) desc limit 15;" \
  /tmp/candle-metal-profile-llm.json
```

## Embedding the profiler in your own code

```rust
use candle_core::{Device, Tensor};

let device = Device::new_metal(0)?;
let metal = match &device {
    Device::Metal(m) => m,
    _ => unreachable!(),
};

metal.install_profiler()?;
// ... run your model ...
metal.flush_profile("/tmp/my-profile.json")?;
```

`install_profiler` validates that the device supports the timestamp counter
set at stage boundaries (every Apple Silicon Mac does). `flush_profile` drains
pending command buffers, then writes the trace. For long-running services,
prefer `flush_profile_and_clear` to avoid unbounded event growth.

## What the trace contains

Each event lands on one of six lanes:

| Lane | Captures |
| --- | --- |
| `metal-gpu-encoder` | Per-encoder GPU duration via stage-boundary counter sampling, with bindings, dispatch grid, pipeline state, and threadgroup memory in args |
| `metal-gpu-cmdbuf` | Per-command-buffer GPU rollup with `gpuStartTime`/`gpuEndTime`, `kernelStartTime`/`kernelEndTime`, scheduling latency, encoder count, error |
| `metal-cpu-encode` | CPU wall time bracketing each encoder (construction → `endEncoding`) |
| `metal-cpu-dispatch` | CPU wall time around each `dispatchThreads*` call |
| `metal-cpu-alloc` | CPU buffer-allocation events with `pool`, `storage`, `requested_bytes`, `rounded_bytes`, `reused` |
| `metal-cpu-sync` | `commit` and `waitUntilCompleted` wall time, with status transitions |

Every encoder event carries `command_buffer_profile_id` + `encoder_index` so
SQL queries can correlate per-encoder GPU duration to its CPU encode time and
the parent command buffer's rollup.

## When to reach for `.gputrace` or `xctrace` instead

The Candle JSON profiler is for measuring **changes** to candle's GPU work —
how a kernel optimization changes per-kernel GPU duration, how a buffer-pool
tweak changes allocation patterns, how a scheduling change shifts
command-buffer residency. It is intentionally Candle-aware, not a replacement
for Apple's tools:

- For full Metal command-stream replay, shader debugging, and resource
  inspection, use **Xcode GPU Frame Capture** (`MetalDevice::capture()` →
  `.gputrace`).
- For system-level driver intervals, resource allocation tables, and
  Instruments counters, use **Instruments / `xctrace record --template
  'Metal System Trace'`** → `.trace`.

The detailed guide at
[`candle-metal-kernels/METAL_PROFILING.md`](https://github.com/huggingface/candle/blob/main/candle-metal-kernels/METAL_PROFILING.md)
covers all three workflows side-by-side, with concrete `xctrace export` XML
schemas, a partial `.gputrace` bundle inspector, and explicit
"Boundaries and anti-proofs" for what each format can and cannot reproduce.
