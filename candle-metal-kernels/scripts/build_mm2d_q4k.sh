#!/bin/sh
# Rebuild the prebuilt tensor-op metallib (mm2d_q4k.metallib) from
# mm2d_q4k.metal. Needs a macOS 26.4+ SDK with the MetalPerformancePrimitives
# framework and the Metal Toolchain component (xcodebuild -downloadComponent
# MetalToolchain) — the runtime compiler cannot build this source, which is
# why the binary is checked in. Run on any capable machine and commit the
# result; metallibs are AIR archives, portable across Apple GPUs.
set -eu
cd "$(dirname "$0")/../src/metal_src"
xcrun metal -std=metal4.0 -c mm2d_q4k.metal -o /tmp/mm2d_q4k.air
xcrun metallib /tmp/mm2d_q4k.air -o mm2d_q4k.metallib
rm /tmp/mm2d_q4k.air
echo "rebuilt $(pwd)/mm2d_q4k.metallib"
