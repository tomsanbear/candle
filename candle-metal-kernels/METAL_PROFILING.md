# Profiling Candle on Metal

This guide covers three complementary workflows for profiling Candle's Metal backend:

1. Candle's structured Chrome/Perfetto JSON profiler.
2. Apple's Xcode GPU Frame Capture (`.gputrace`).
3. Apple's Instruments / `xctrace` Metal System Trace (`.trace`).

These tools answer different questions. The Candle JSON profiler is portable and Candle-aware: it records kernel labels, command-buffer timing, CPU encode/dispatch/allocation/commit/wait lanes, and public Metal counter metadata. Apple's tools are the source of truth for Xcode/Instruments-only driver, resource, shader, and system-level timelines.

## Build prerequisites

Use the `metal-profile` feature on crates/examples that use Candle's Metal backend.

```bash
cargo build -p candle-core --example metal_profile --features metal-profile
cargo build -p candle-examples --example metal_profile_llm --features metal-profile --release
```

The commands below were validated with Xcode 26 `xcrun xctrace` on an Apple M4 Max. `xctrace` flags vary between Xcode versions; on this setup `--target-stdout` is supported but `--target-stderr` is not. If your `xctrace record --help` does not list `--target-stdout`, omit that flag.

## Demo scripts

The `tools/metal-profiling/` directory contains small scripts that demonstrate the workflows and the format boundaries described in this guide:

- `run_candle_metal_profile_demo.sh` — builds an example, writes Candle JSON, writes `.gputrace`, records an Instruments `.trace`, exports useful XML tables, and prints summaries.
- `parse_xctrace_table.py` — parses one `xctrace export` XML table, resolving `id`/`ref` nodes and using schema `mnemonic` names as columns.
- `inspect_gputrace_bundle.py` — inspects the public/observable parts of a `.gputrace` bundle: metadata, raw `MTLBuffer-*`/`MTLTexture-*` files if present, `index`/`storeN` shape, and printable strings.
- `check_metal_profile_boundaries.sh` — demonstrates the current public/tooling boundaries, including that `xctrace export` does not export `.gputrace` bundles.

Run the complete small demo:

```bash
candle-metal-kernels/tools/metal-profiling/run_candle_metal_profile_demo.sh
```

Run the LLM-shaped demo with a short decode:

```bash
EXAMPLE=llm DECODE_STEPS=4 \
  candle-metal-kernels/tools/metal-profiling/run_candle_metal_profile_demo.sh
```

Then run the boundary checks:

```bash
candle-metal-kernels/tools/metal-profiling/check_metal_profile_boundaries.sh
```

By default, artifacts are written under `/tmp/candle-metal-profile-demo`.

## 1. Candle JSON profiler

Run the small synthetic Metal profile example:

```bash
cargo run -p candle-core --example metal_profile --features metal-profile -- \
  --profile-json /tmp/candle-metal-profile.json
```

Run the self-contained Llama-shaped decode profile:

```bash
cargo run -p candle-examples --example metal_profile_llm --release \
  --features metal-profile -- \
  --decode-steps 32 \
  --profile-json /tmp/candle-metal-profile-llm.json
```

Open the JSON in <https://ui.perfetto.dev/> or query it with Perfetto trace processor:

```bash
trace_processor -Q \
  "select cat, count(*) n, round(sum(dur)/1000.0, 3) total_us \
   from slice group by cat order by n desc;" \
  /tmp/candle-metal-profile.json

trace_processor -Q \
  "select name, count(*) n, round(sum(dur)/1000.0, 3) total_us \
   from slice where cat = 'metal-gpu-encoder' \
   group by name order by sum(dur) desc limit 15;" \
  /tmp/candle-metal-profile.json
```

The JSON uses async begin/end events so overlapping Metal samples are preserved without creating one lane per event. Current lanes are:

- GPU encoders
- GPU command buffers
- CPU encode
- CPU dispatch
- CPU buffer allocation/reuse
- CPU commit/wait

The profiler also writes metadata for public Metal counter-set discovery, supported public sampling points, sample-buffer capacity, and any dropped encoder-profile attempts. On the validated M4 Max, public `MTLDevice.counterSets` exposed only the timestamp counter set, with stage-boundary sampling supported and draw/dispatch/tile-dispatch/blit boundary sampling not reported as supported:

```json
{
  "name": "timestamp",
  "counters": ["GPUTimestamp"]
}
```

Do not assume statistic or stage-utilization counters, or a specific sampling point, are available through public Metal APIs unless the trace metadata says they are present on the target device/OS. Each encoder consumes two timestamp sample slots; if a command buffer exceeds the sample capacity, the encoder still runs but `dropped_encoder_profiles` is incremented in trace metadata.

`flush_profile` writes the current in-memory event sink but intentionally keeps events for snapshots or repeated writes. For periodic profiling in long-running processes, use `flush_profile_and_clear` or call `profile_clear` after writing.

## 2. Xcode GPU Frame Capture (`.gputrace`)

Candle examples can start/stop an Apple Metal capture around the profiled workload:

```bash
MTL_CAPTURE_ENABLED=1 cargo run -p candle-core --example metal_profile \
  --features metal-profile -- \
  --gputrace /tmp/candle-metal-profile.gputrace \
  --profile-json /tmp/candle-metal-profile.json
```

For the LLM-shaped example:

```bash
MTL_CAPTURE_ENABLED=1 cargo run -p candle-examples --example metal_profile_llm --release \
  --features metal-profile -- \
  --decode-steps 4 \
  --gputrace /tmp/candle-metal-profile-llm.gputrace \
  --profile-json /tmp/candle-metal-profile-llm.json
```

Open the capture in Xcode:

```bash
open -a Xcode /tmp/candle-metal-profile.gputrace
```

Notes:

- `MTL_CAPTURE_ENABLED=1` is required when running outside Xcode.
- Delete any existing output bundle first. Apple's capture API fails rather than overwriting an existing `.gputrace` path.
- `.gputrace` is an Xcode GPU Frame Capture package, not an Instruments `.trace` package. `xcrun xctrace export --input some.gputrace --toc` may fail with `Document Missing Template Error`.
- Current Xcode `.gputrace` bundles are private packages. Locally observed bundles contain a binary-plist `metadata` file, an `index` file beginning with `xdic`, compressed `storeN` payloads, and sometimes raw `MTLBuffer-*` / `MTLTexture-*` resource files. Shared-storage buffers from simple examples may appear as raw files; Candle tensors are commonly `StorageModePrivate`, so raw buffer dumps may be absent even though Xcode can still open the capture.
- When capture is enabled, Apple may wrap the Metal device in a `CaptureMTLDevice` proxy. Candle's profiler treats some device metadata as optional in that mode to avoid calling selectors that the proxy does not implement.

## 3. Instruments / `xctrace` Metal System Trace (`.trace`)

Record an Instruments Metal System Trace while launching a Candle example:

```bash
xcrun xctrace record \
  --template 'Metal System Trace' \
  --output /tmp/candle-metal-system.trace \
  --target-stdout /tmp/xctrace-target.stdout \
  --launch -- \
  ${CARGO_TARGET_DIR:-target}/debug/examples/metal_profile \
  --profile-json /tmp/xctrace-candle-profile.json
```

For the release LLM-shaped example:

```bash
xcrun xctrace record \
  --template 'Metal System Trace' \
  --output /tmp/candle-metal-llm-system.trace \
  --target-stdout /tmp/xctrace-llm.stdout \
  --launch -- \
  ${CARGO_TARGET_DIR:-target}/release/examples/metal_profile_llm \
  --decode-steps 4 \
  --profile-json /tmp/xctrace-llm-profile.json
```

Inspect the trace table of contents:

```bash
xcrun xctrace export --input /tmp/candle-metal-system.trace --toc > /tmp/candle-metal-system-toc.xml
```

`xctrace` XML exports include a schema block with column `mnemonic`s followed by rows whose elements use engineering-type tag names. Repeated values are emitted once with `id="..."` and referenced later with `ref="..."`, so parsers must resolve refs before interpreting rows.

Export selected tables:

```bash
xcrun xctrace export \
  --input /tmp/candle-metal-system.trace \
  --output /tmp/metal-application-encoders-list.xml \
  --xpath "/trace-toc/run[@number='1']/data/table[@schema='metal-application-encoders-list']"

xcrun xctrace export \
  --input /tmp/candle-metal-system.trace \
  --output /tmp/metal-resource-allocations.xml \
  --xpath "/trace-toc/run[@number='1']/data/table[@schema='metal-resource-allocations']"
```

Minimal parser pattern for one exported table:

```python
import xml.etree.ElementTree as ET
from collections import Counter

root = ET.parse('/tmp/metal-application-encoders-list.xml').getroot()
id_map = {e.get('id'): e for e in root.iter() if e.get('id')}

def value(e):
    if e.get('ref') in id_map:
        e = id_map[e.get('ref')]
    return e.get('fmt', e.text or '')

schema = root.find('.//schema')
cols = [c.findtext('mnemonic') for c in schema.findall('col')]
rows = []
for row in root.findall('.//row'):
    rows.append({cols[i]: value(c) for i, c in enumerate(list(row))})

print(Counter(r['encoder-label'] for r in rows).most_common())
```

Useful schemas observed in Metal System Trace include:

- `metal-application-encoders-list` — encoder labels, command-buffer labels, encoder ids, CPU encoding intervals.
- `metal-application-command-buffer-submissions` — command-buffer submission summaries.
- `metal-command-buffer-completed` — command-buffer completion records.
- `metal-gpu-intervals` — GPU execution intervals, including system activity.
- `metal-gpu-info` and `device-gpu-info` — GPU/device metadata.
- `gpu-counter-info`, `gpu-counter-value`, and `metal-gpu-counter-intervals` — Instruments counter streams when present.
- `metal-resource-allocations` and `metal-current-allocated-size` — resource allocation/deallocation and process allocation level.
- `metal-driver-event-intervals` and `metal-driver-intervals` — driver work such as wire/unwire memory events.
- `metal-object-label` — labels captured from Metal objects. This is useful for checking that Candle labels are visible to Apple's tooling.

On the validated M4 Max, adding the `Metal GPU Counters` instrument selected `Counter Set: Performance Limiters` but produced this warning:

```text
GPU Service reported error: Selected counter profile is not supported on target device
```

So treat Instruments counter availability as device/OS/template-specific and verify it locally before documenting or relying on any specific counter set.

## What the output formats are, and are not

The three outputs overlap in concepts but are not interchangeable:

| Output | Created by | CLI-parseable? | Best use |
| --- | --- | --- | --- |
| Candle JSON | Candle profiler | Yes | Candle-native kernel labels, public timestamp timing, CPU lanes, buffer allocation/reuse metadata |
| `.gputrace` | `MTLCaptureManager` / Xcode GPU Capture | Partially/unofficially | Xcode GPU debugger, command/resource/pipeline inspection in Xcode |
| `.trace` | Instruments / `xcrun xctrace record` | Yes, via `xctrace export` XML | System-level Metal timelines, driver intervals, resource allocation tables, object labels, Instruments counters when available |

Important distinction: `.gputrace` and `.trace` are different Apple package formats. `xctrace export` works on Instruments `.trace` bundles, not Xcode GPU Capture `.gputrace` bundles.

## Boundaries and anti-proofs

These are the current public/stable API boundaries validated during development. They are written negatively on purpose so future changes do not accidentally overclaim.

### 1. Full `.gputrace` CLI parsing is not public/stable

Best-effort anti-proof: `.gputrace` is not completely opaque. The observable bundle can include:

- `metadata`, a binary plist parseable with `plistlib` or `plutil`.
- `index`, locally observed with an `xdic` header.
- `storeN`, locally observed as zlib-compressed private payloads.
- Sometimes raw `MTLBuffer-*` / `MTLTexture-*` files.

Use:

```bash
python3 candle-metal-kernels/tools/metal-profiling/inspect_gputrace_bundle.py \
  /tmp/candle-metal-profile-demo/candle-metal-profile.gputrace --strings
```

Boundary: no public stable CLI/API was found that reconstructs the full Xcode GPU debugger model from `.gputrace` — dispatch/draw call stepping, full pipeline state, shader debugger state, pixel history, and Xcode's resource views remain Xcode/private tooling.

### 2. Candle JSON cannot recreate all Xcode/Instruments data

Candle can record or approximate many useful app-level fields with public APIs:

- Encoder labels and command-buffer labels.
- Per-encoder public timestamp samples.
- Command-buffer completion timing.
- CPU encode/dispatch/allocation/commit/wait intervals.
- Pipeline metadata available from `MTLComputePipelineState`.
- Dispatch dimensions and buffer binding metadata tracked while encoding.
- Device metadata and public counter-set enumeration.

Boundary: Xcode/Instruments also records private/tooling-derived data such as driver intervals, wire/unwire memory events, shader profiler internals, object dependency chains, allocation backtraces, system GPU state, and private counter streams. Candle should correlate with those outputs, not claim to replace them.

### 3. Public Metal counters cannot be forced

Public code can enumerate:

```rust
let counter_sets = device.counterSets();
let supports_stage_boundary =
    device.supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary);
```

Boundary: no public Metal API, environment variable, or `xctrace` flag was found that makes `MTLDevice.counterSets` expose counter sets the device/OS does not report. Instruments counter profiles may use Apple-internal paths; Candle must treat public counter availability as runtime data.

### 4. Raw `MTLBuffer-*` / `MTLTexture-*` files in `.gputrace` are not guaranteed

Best-effort anti-proof: simple captures with `StorageModeShared` buffers can produce raw `MTLBuffer-*` files that are readable from the command line. If such a file exists:

```bash
python3 candle-metal-kernels/tools/metal-profiling/inspect_gputrace_bundle.py \
  capture.gputrace --buffer MTLBuffer-14-0 --layout float4 --index 0-5
```

Boundary: Candle tensors are commonly `StorageModePrivate` on macOS for performance. Private resources often do not appear as raw root `MTLBuffer-*` files even though Xcode can open the capture. Labels help Xcode and Instruments, but no public `MTLCaptureDescriptor` option was found that guarantees raw resource-file emission.

### 5. `xctrace export` does not export `.gputrace`

Demonstration:

```bash
xcrun xctrace export \
  --input /tmp/candle-metal-profile-demo/candle-metal-profile.gputrace \
  --toc
```

Locally this fails with `Document Missing Template Error`. `xctrace import`/`remodel` also target Instruments-supported formats, not `.gputrace` GPU Capture packages.

### 6. Shader step-through, pixel history, and full debugger UI are not public CLI features

Public/CLI workflows do exist for adjacent tasks:

- `MTLCaptureManager` can write `.gputrace`.
- `MTL_DEBUG_LAYER=1` and `MTL_SHADER_VALIDATION=1` enable validation.
- `xcrun xctrace record` records Metal System Trace.
- `xcrun metal` compiles and diagnoses shaders.

Boundary: shader step-through, pixel history, and full pipeline/resource debugger inspection are Xcode GPU debugger features, not public stable command-line APIs.

### 7. Rich Apple GPU performance counters are not portable

The portable pattern is runtime enumeration plus metadata in the output trace. On the validated M4 Max, public Metal exposed only the timestamp counter set:

```json
{
  "name": "timestamp",
  "counters": ["GPUTimestamp"]
}
```

Boundary: do not claim statistic, stage-utilization, performance-limiter, occupancy, bandwidth, or other rich counters across Apple GPU generations unless the current device/OS reports them at runtime.

## Interpreting the three outputs together

- Use Candle JSON to answer Candle-native questions: which kernel labels ran, how long public Metal timestamp samples measured, CPU-side encode/dispatch/allocation/commit/wait overhead, and whether buffers were allocated or reused.
- Use `.gputrace` to inspect the workload in Xcode's GPU capture UI.
- Use `.trace` / `xctrace` to inspect Instruments tables for command-buffer submission, resource allocation, driver intervals, device info, and system GPU activity.

Candle's JSON profiler intentionally does not claim to reproduce all Xcode GPU trace information. Some Xcode/Instruments data is produced by Apple's capture and driver tooling and is not exposed through stable public Metal APIs.