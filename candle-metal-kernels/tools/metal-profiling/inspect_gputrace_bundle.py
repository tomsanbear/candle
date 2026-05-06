#!/usr/bin/env python3
"""Inspect the public/observable parts of an Xcode `.gputrace` bundle.

This is intentionally a partial inspector. `.gputrace` is a private Xcode GPU
capture format. Some captures contain raw `MTLBuffer-*` / `MTLTexture-*` files
that can be read from the command line; many captures, especially those using
`StorageModePrivate` resources, do not.

Examples:
    python3 inspect_gputrace_bundle.py /tmp/candle-metal-profile.gputrace
    python3 inspect_gputrace_bundle.py capture.gputrace --buffer MTLBuffer-14-0 --layout float4 --index 0-5
    python3 inspect_gputrace_bundle.py capture.gputrace --strings
"""

from __future__ import annotations

import argparse
import json
import plistlib
import re
import struct
import zlib
from pathlib import Path


def parse_metadata(path: Path) -> dict:
    meta = path / "metadata"
    if not meta.exists():
        return {}
    with meta.open("rb") as f:
        return plistlib.load(f)


def printable_strings(data: bytes, min_len: int = 4) -> list[str]:
    out: list[str] = []
    cur: list[int] = []
    for b in data:
        if 32 <= b <= 126:
            cur.append(b)
        else:
            if len(cur) >= min_len:
                out.append(bytes(cur).decode("utf-8", "replace"))
            cur = []
    if len(cur) >= min_len:
        out.append(bytes(cur).decode("utf-8", "replace"))
    return out


def try_decompress(data: bytes) -> bytes | None:
    for wbits in (zlib.MAX_WBITS, -zlib.MAX_WBITS, zlib.MAX_WBITS | 32):
        try:
            return zlib.decompress(data, wbits)
        except zlib.error:
            pass
    return None


def parse_layout(layout: str) -> tuple[str, list[tuple[str, int]]]:
    fmt = "<"
    fields: list[tuple[str, int]] = []
    for part in layout.split(","):
        part = part.strip()
        if part == "float":
            fmt += "f"; fields.append((part, 1))
        elif part == "float2":
            fmt += "ff"; fields.append((part, 2))
        elif part == "float3":
            fmt += "fff"; fields.append((part, 3))
        elif part == "float4":
            fmt += "ffff"; fields.append((part, 4))
        elif part == "uint32":
            fmt += "I"; fields.append((part, 1))
        elif part == "int32":
            fmt += "i"; fields.append((part, 1))
        else:
            raise ValueError(f"unsupported layout component: {part}")
    return fmt, fields


def read_buffer(path: Path, layout: str, index: str) -> list[dict]:
    if "-" in index:
        start_s, end_s = index.split("-", 1)
        start, end = int(start_s), int(end_s)
    else:
        start = end = int(index)
    fmt, fields = parse_layout(layout)
    stride = struct.calcsize(fmt)
    data = path.read_bytes()
    total = len(data) // stride
    rows = []
    for idx in range(start, min(end + 1, total)):
        values = struct.unpack_from(fmt, data, idx * stride)
        row = {"index": idx}
        cursor = 0
        for field_i, (_, n) in enumerate(fields):
            vals = values[cursor : cursor + n]
            row[f"field{field_i}"] = vals[0] if n == 1 else list(vals)
            cursor += n
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gputrace", type=Path)
    parser.add_argument("--buffer", help="raw MTLBuffer-* filename to read")
    parser.add_argument("--layout", default="float4")
    parser.add_argument("--index", default="0-10")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strings", action="store_true", help="print strings from index/store payloads")
    args = parser.parse_args()

    root = args.gputrace
    if not root.is_dir():
        raise SystemExit(f"not a .gputrace directory: {root}")

    if args.buffer:
        buf = root / args.buffer
        if not buf.exists():
            raise SystemExit(f"buffer file not found: {buf}")
        rows = read_buffer(buf, args.layout, args.index)
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            print(f"{args.buffer} layout={args.layout} index={args.index}")
            for row in rows:
                fields = [v for k, v in row.items() if k != "index"]
                print(f"  [{row['index']:>6}] {fields}")
        return 0

    meta = parse_metadata(root)
    print(f"GPU trace: {root}")
    if meta:
        print(f"  UUID: {meta.get('(uuid)', 'unknown')}")
        print(f"  Frames: {meta.get('DYCaptureEngine.captured_frames_count', '?')}")
        print(f"  API: {meta.get('DYCaptureSession.graphics_api', '?')} (1 = Metal)")
        print(f"  Capture version: {meta.get('DYCaptureSession.capture_version', '?')}")
    print()

    files = sorted(p for p in root.iterdir() if p.is_file())
    raw_resources = [p for p in files if re.match(r"MTL(Buffer|Texture)-\d+-\d+", p.name)]
    if raw_resources:
        print("Raw resource files visible to CLI:")
        for p in raw_resources:
            print(f"  {p.name:28s} {p.stat().st_size:12,} bytes")
    else:
        print("Raw resource files visible to CLI: none")
        print("  Note: this is common for captures dominated by StorageModePrivate buffers.")
    print()

    print("Internal files:")
    for p in files:
        if p in raw_resources or p.name == "metadata":
            continue
        head = p.read_bytes()[:8].hex()
        desc = ""
        if p.name.startswith("store"):
            data = p.read_bytes()
            decomp = try_decompress(data)
            if decomp is not None:
                desc = f" zlib -> {len(decomp):,} bytes"
        elif p.name == "index":
            data = p.read_bytes()
            if data.startswith(b"xdic"):
                desc = " xdic"
        print(f"  {p.name:28s} {p.stat().st_size:12,} bytes head={head}{desc}")

    if args.strings:
        print("\nPrintable strings from index/store payloads:")
        seen: set[str] = set()
        for p in files:
            if p.name == "metadata" or p in raw_resources:
                continue
            data = p.read_bytes()
            decomp = try_decompress(data) if p.name.startswith("store") else None
            for source, payload in [(p.name, data), (p.name + " (decompressed)", decomp)]:
                if payload is None:
                    continue
                strings = [s for s in printable_strings(payload) if s not in seen]
                for s in strings[:100]:
                    seen.add(s)
                    print(f"  {source}: {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
