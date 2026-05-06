#!/usr/bin/env python3
"""Parse one `xcrun xctrace export` XML table.

`xctrace export` writes rows whose element names are engineering types
(`start-time`, `duration`, `metal-object-label`, ...), not the logical column
names. The logical names are in the table schema as `mnemonic` values. Repeated
values are de-duplicated with `id` / `ref`, so refs must be resolved.

Examples:
    xcrun xctrace export \
      --input /tmp/candle-metal-system.trace \
      --output /tmp/encoders.xml \
      --xpath "/trace-toc/run[@number='1']/data/table[@schema='metal-application-encoders-list']"

    python3 parse_xctrace_table.py /tmp/encoders.xml --summary
    python3 parse_xctrace_table.py /tmp/encoders.xml --json --limit 5
    python3 parse_xctrace_table.py /tmp/encoders.xml --count encoder-label
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


def load_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    root = ET.parse(path).getroot()
    id_map = {elem.get("id"): elem for elem in root.iter() if elem.get("id")}

    def resolved(elem: ET.Element) -> ET.Element:
        ref = elem.get("ref")
        if ref is not None and ref in id_map:
            return id_map[ref]
        return elem

    def value(elem: ET.Element) -> str:
        elem = resolved(elem)
        return elem.get("fmt", elem.text or "")

    schema = root.find(".//schema")
    headers: list[str] = []
    if schema is not None:
        for i, col in enumerate(schema.findall("col")):
            headers.append(
                col.findtext("mnemonic")
                or col.findtext("name")
                or col.findtext("engineering-type")
                or f"col_{i}"
            )

    rows: list[dict[str, str]] = []
    for row in root.findall(".//row"):
        cols = list(row)
        if not headers:
            headers = [f"col_{i}" for i in range(len(cols))]
        rows.append(
            {
                headers[i] if i < len(headers) else f"col_{i}": value(col)
                for i, col in enumerate(cols)
            }
        )
    return headers, rows


def print_summary(headers: list[str], rows: list[dict[str, str]]) -> None:
    print(f"rows: {len(rows)}")
    print(f"columns: {', '.join(headers)}")
    for col in headers:
        values = [row.get(col, "") for row in rows]
        unique = len(set(values))
        if unique <= 1 or col in {
            "event-type",
            "event-label",
            "encoder-label",
            "cmdbuffer-label",
            "process",
            "gpu",
            "label",
            "object-type",
            "resource-type",
        }:
            print(f"\n{col}: {unique} unique")
            for value, count in Counter(values).most_common(12):
                print(f"  {count:6d}  {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml", type=Path, help="xctrace-exported XML table")
    parser.add_argument("--summary", action="store_true", help="show row/column summary")
    parser.add_argument("--json", action="store_true", help="emit rows as JSON")
    parser.add_argument("--csv", action="store_true", help="emit rows as CSV")
    parser.add_argument("--count", help="count values in a column mnemonic")
    parser.add_argument("--limit", type=int, default=50, help="max rows for JSON/CSV/TSV")
    args = parser.parse_args()

    headers, rows = load_table(args.xml)
    if args.summary:
        print_summary(headers, rows)
    elif args.count:
        if args.count not in headers:
            print(f"unknown column {args.count!r}; available: {', '.join(headers)}", file=sys.stderr)
            return 2
        for value, count in Counter(row.get(args.count, "") for row in rows).most_common():
            print(f"{count}\t{value}")
    elif args.json:
        print(json.dumps(rows[: args.limit], indent=2))
        if len(rows) > args.limit:
            print(f"// ... {len(rows) - args.limit} more rows", file=sys.stderr)
    elif args.csv:
        writer = csv.DictWriter(sys.stdout, fieldnames=headers)
        writer.writeheader()
        for row in rows[: args.limit]:
            writer.writerow(row)
    else:
        print("\t".join(headers))
        for row in rows[: args.limit]:
            print("\t".join(row.get(header, "") for header in headers))
        if len(rows) > args.limit:
            print(f"... {len(rows) - args.limit} more rows", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
