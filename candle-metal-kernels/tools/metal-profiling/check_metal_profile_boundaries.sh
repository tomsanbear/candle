#!/usr/bin/env bash
# Demonstrate the current public/tooling boundaries around Candle Metal traces.
# Run this after `run_candle_metal_profile_demo.sh` or set OUT_DIR to the demo output.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT_DIR="${OUT_DIR:-/tmp/candle-metal-profile-demo}"
GPUTRACE="$OUT_DIR/candle-metal-profile.gputrace"
TRACE="$OUT_DIR/candle-metal-system.trace"
JSON="$OUT_DIR/candle-metal-profile.json"
PARSER="$ROOT/candle-metal-kernels/tools/metal-profiling/parse_xctrace_table.py"
INSPECTOR="$ROOT/candle-metal-kernels/tools/metal-profiling/inspect_gputrace_bundle.py"

failures=0

section() { printf '\n== %s ==\n' "$*"; }

if [[ ! -d "$GPUTRACE" || ! -d "$TRACE" || ! -f "$JSON" ]]; then
  cat >&2 <<EOF
Missing demo artifacts under $OUT_DIR.
Run first:
  OUT_DIR="$OUT_DIR" candle-metal-kernels/tools/metal-profiling/run_candle_metal_profile_demo.sh
Expected:
  $GPUTRACE
  $TRACE
  $JSON
EOF
  exit 2
fi

section ".gputrace is not an xctrace-exportable .trace"
if xcrun xctrace export --input "$GPUTRACE" --toc >/tmp/candle-gputrace-toc.out 2>/tmp/candle-gputrace-toc.err; then
  echo "UNEXPECTED: xctrace exported .gputrace TOC"
  failures=$((failures + 1))
else
  echo "expected failure: $(tr '\n' ' ' </tmp/candle-gputrace-toc.err | sed 's/[[:space:]]*$//')"
fi

section ".gputrace observable bundle contents"
python3 "$INSPECTOR" "$GPUTRACE"

section "Candle JSON public counter metadata"
python3 - <<PY
import json
path = "$JSON"
with open(path) as f:
    trace = json.load(f)
for event in trace.get('traceEvents', []):
    if event.get('name') == 'candle_metal_counter_sets':
        print(json.dumps(event.get('args', {}), indent=2))
        break
else:
    print('missing candle_metal_counter_sets metadata')
    raise SystemExit(1)
PY

section "xctrace XML encoder labels"
ENCODERS="$OUT_DIR/xctrace/metal-application-encoders-list.xml"
if [[ -f "$ENCODERS" ]]; then
  python3 "$PARSER" "$ENCODERS" --count encoder-label | head -40
else
  echo "missing $ENCODERS; run run_candle_metal_profile_demo.sh first"
  failures=$((failures + 1))
fi

section "xctrace resource allocation labels"
RES="$OUT_DIR/xctrace/metal-resource-allocations.xml"
if [[ -f "$RES" ]]; then
  python3 "$PARSER" "$RES" --count label | head -40
else
  echo "missing $RES; run run_candle_metal_profile_demo.sh first"
fi

section "Boundary verdicts"
cat <<'EOF'
1. Full .gputrace CLI parsing remains unavailable through public stable APIs.
   Partial metadata/raw shared-resource inspection is possible.
2. Candle JSON can approximate many useful app-level fields, but not all
   Xcode/Instruments private driver/debugger data.
3. Public Metal counter sets cannot be forced beyond runtime enumeration.
4. Raw MTLBuffer-/MTLTexture-* files in .gputrace cannot be guaranteed;
   StorageModeShared resources may appear, StorageModePrivate resources often do not.
5. xctrace export targets Instruments .trace, not Xcode .gputrace.
6. Shader step-through, pixel history, and full debugger UI remain Xcode/private GUI territory.
7. Rich Apple GPU performance counters are not portable; enumerate at runtime.
EOF

exit "$failures"
