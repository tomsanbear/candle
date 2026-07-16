#!/bin/sh
# Rebuild the prebuilt tensor-op metallib (mm2d_q2_0.metallib) from
# mm2d_q2_0.metal. Needs the Metal 4.1 toolchain (uint2b_format matmul2d
# operand) — shipped in Xcode 27 beta 3's Metal Toolchain component
# (metalfe-32023.918), NOT in Xcode 26.6 (32023.883). Point at the beta and
# pull its Metal Toolchain first:
#   DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer \
#     xcodebuild -downloadComponent MetalToolchain
# The runtime compiler cannot build this source (framework header), which is
# why the binary is checked in. metallibs are AIR archives, portable across
# Apple GPUs.
set -eu
: "${DEVELOPER_DIR:=/Applications/Xcode-beta.app/Contents/Developer}"
export DEVELOPER_DIR
cd "$(dirname "$0")/../src/metal_src"
xcrun metal -std=metal4.1 -c mm2d_q2_0.metal -o /tmp/mm2d_q2_0.air
xcrun metallib /tmp/mm2d_q2_0.air -o mm2d_q2_0.metallib
rm /tmp/mm2d_q2_0.air
echo "rebuilt $(pwd)/mm2d_q2_0.metallib"
