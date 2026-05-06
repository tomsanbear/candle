#!/usr/bin/env bash
# Run Candle's three Metal profiling workflows and export parseable artifacts.
#
# Outputs under /tmp/candle-metal-profile-demo by default:
#   candle-metal-profile.json       Candle-native Chrome/Perfetto JSON
#   candle-metal-profile.gputrace   Xcode GPU Frame Capture bundle
#   candle-metal-system.trace       Instruments Metal System Trace bundle
#   xctrace/*.xml                   Exported Instruments tables
#
# Usage:
#   candle-metal-kernels/tools/metal-profiling/run_candle_metal_profile_demo.sh
#   OUT_DIR=/tmp/my-demo .../run_candle_metal_profile_demo.sh
#   EXAMPLE=llm DECODE_STEPS=4 .../run_candle_metal_profile_demo.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT_DIR="${OUT_DIR:-/tmp/candle-metal-profile-demo}"
EXAMPLE="${EXAMPLE:-small}" # small | llm
DECODE_STEPS="${DECODE_STEPS:-4}"
PARSER="$ROOT/candle-metal-kernels/tools/metal-profiling/parse_xctrace_table.py"
GPUTRACE_INSPECTOR="$ROOT/candle-metal-kernels/tools/metal-profiling/inspect_gputrace_bundle.py"

rm -rf "$OUT_DIR/xctrace"
mkdir -p "$OUT_DIR/xctrace"
rm -rf "$OUT_DIR/candle-metal-profile.gputrace" "$OUT_DIR/candle-metal-system.trace"

cd "$ROOT"

if [[ "$EXAMPLE" != "small" && "$EXAMPLE" != "llm" ]]; then
  echo "EXAMPLE must be 'small' or 'llm' (got: $EXAMPLE)" >&2
  exit 2
fi

if [[ "$EXAMPLE" == "llm" ]]; then
  echo "== building LLM-shaped example =="
  cargo build -p candle-examples --example metal_profile_llm --release --features metal-profile
  TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
  BIN="$TARGET_DIR/release/examples/metal_profile_llm"
  TARGET_ARGS=(--decode-steps "$DECODE_STEPS" --profile-json "$OUT_DIR/candle-metal-profile.json")
else
  echo "== building small profile example =="
  cargo build -p candle-core --example metal_profile --features metal-profile
  TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
  BIN="$TARGET_DIR/debug/examples/metal_profile"
  TARGET_ARGS=(--profile-json "$OUT_DIR/candle-metal-profile.json")
fi

printf '\n== Candle JSON only ==\n'
"$BIN" "${TARGET_ARGS[@]}"

printf '\n== Xcode .gputrace capture ==\n'
rm -rf "$OUT_DIR/candle-metal-profile.gputrace"
MTL_CAPTURE_ENABLED=1 "$BIN" \
  --gputrace "$OUT_DIR/candle-metal-profile.gputrace" \
  "${TARGET_ARGS[@]}"
python3 "$GPUTRACE_INSPECTOR" "$OUT_DIR/candle-metal-profile.gputrace" || true

printf '\n== Instruments Metal System Trace ==\n'
rm -rf "$OUT_DIR/candle-metal-system.trace"
xctrace_cmd=(
  xcrun xctrace record
  --template 'Metal System Trace'
  --output "$OUT_DIR/candle-metal-system.trace"
)
if xcrun xctrace record --help 2>&1 | grep -q -- '--target-stdout'; then
  xctrace_cmd+=(--target-stdout "$OUT_DIR/xctrace-target.stdout")
fi
xctrace_cmd+=(--launch -- "$BIN" "${TARGET_ARGS[@]}")
"${xctrace_cmd[@]}"

xcrun xctrace export --input "$OUT_DIR/candle-metal-system.trace" --toc \
  > "$OUT_DIR/xctrace/toc.xml"

schemas=(
  metal-application-encoders-list
  metal-application-command-buffer-submissions
  metal-gpu-intervals
  metal-driver-event-intervals
  metal-driver-intervals
  metal-resource-allocations
  metal-current-allocated-size
  metal-object-label
  gpu-counter-info
  gpu-counter-value
  metal-gpu-counter-intervals
  device-gpu-info
  metal-gpu-info
)

printf '\n== Exporting xctrace tables ==\n'
for schema in "${schemas[@]}"; do
  if grep -q "schema=\"$schema\"" "$OUT_DIR/xctrace/toc.xml"; then
    echo "export $schema"
    xcrun xctrace export \
      --input "$OUT_DIR/candle-metal-system.trace" \
      --output "$OUT_DIR/xctrace/$schema.xml" \
      --xpath "/trace-toc/run[@number='1']/data/table[@schema='$schema']" \
      >/dev/null 2>&1 || echo "  export failed: $schema"
  fi
done

printf '\n== Quick summaries ==\n'
for xml in "$OUT_DIR"/xctrace/*.xml; do
  [[ "$(basename "$xml")" == "toc.xml" ]] && continue
  echo "--- $(basename "$xml")"
  python3 "$PARSER" "$xml" --summary | head -80 || true
done

cat <<EOF

Artifacts written to: $OUT_DIR

Open/inspect:
  open -a Xcode "$OUT_DIR/candle-metal-profile.gputrace"
  open "$OUT_DIR/candle-metal-system.trace"
  python3 "$PARSER" "$OUT_DIR/xctrace/metal-application-encoders-list.xml" --count encoder-label
  python3 "$GPUTRACE_INSPECTOR" "$OUT_DIR/candle-metal-profile.gputrace" --strings
EOF
