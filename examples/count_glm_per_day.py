#!/usr/bin/env python3
"""
Count GOES GLM-L2-LCFA files per day in a public GCS bucket (GOES-16/18/19)
===========================================================================

- Lists objects via the public GCS JSON API (no auth needed)
- Filters to GLM-L2-LCFA files
- Groups counts by (year, jday) parsed from filename (_sYYYYJJJHHMMSS)
- Writes CSV + prints a small summary

Example:
  python count_glm_per_day.py --bucket gcp-public-data-goes-19 --start 2025-10-16 --end 2026-02-01

Notes:
- This counts objects present in the bucket; it does not validate file contents.
- Uses listing prefixes GLM-L2-LCFA/YYYY/JJJ/HH/ to keep requests bounded.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import aiohttp


# ----------------------------
# Parsing helpers
# ----------------------------

_PAT = re.compile(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})")  # _sYYYYJJJHHMMSS

def parse_scan_start(blob_name: str) -> Tuple[int, int] | None:
    """
    Return (year, jday) from the scan start token in the filename, or None.
    """
    m = _PAT.search(blob_name)
    if not m:
        return None
    year = int(m.group(1))
    jday = int(m.group(2))
    return (year, jday)


def date_iter_utc(start_date: str, end_date: str):
    a = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    b = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    cur = a
    while cur <= b:
        yield cur
        cur += timedelta(days=1)


# ----------------------------
# GCS listing (public JSON API)
# ----------------------------

@dataclass(frozen=True)
class Cfg:
    bucket: str
    product: str = "GLM-L2-LCFA"
    timeout_s: int = 60
    list_conc: int = 64
    per_hour_prefix: bool = True  # list 24 hour prefixes/day (faster + smaller pages)


async def gcs_list_names(
    session: aiohttp.ClientSession,
    bucket: str,
    prefix: str,
    sem: asyncio.Semaphore,
    timeout_s: int,
) -> List[str]:
    """
    List object names under a prefix using GCS JSON API with pagination.
    """
    out: List[str] = []
    base = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o"
    page_token: str | None = None
    timeout = aiohttp.ClientTimeout(total=timeout_s)

    # Keep response small
    params = {
        "prefix": prefix,
        "fields": "items(name),nextPageToken",
        "maxResults": "1000",
    }

    while True:
        if page_token:
            params["pageToken"] = page_token
        else:
            params.pop("pageToken", None)

        async with sem:
            try:
                async with session.get(base, params=params, timeout=timeout) as r:
                    if r.status != 200:
                        return out
                    js = await r.json()
            except Exception:
                return out

        for it in js.get("items", []):
            n = it.get("name")
            if n:
                out.append(n)

        page_token = js.get("nextPageToken")
        if not page_token:
            break

    return out


async def list_day_glm(
    session: aiohttp.ClientSession,
    cfg: Cfg,
    day: datetime,
    sem: asyncio.Semaphore,
) -> List[str]:
    """
    List GLM object names for a given UTC day.
    """
    year = day.year
    jday = int(day.strftime("%j"))

    if cfg.per_hour_prefix:
        prefixes = [f"{cfg.product}/{year}/{jday:03d}/{hh:02d}/" for hh in range(24)]
    else:
        prefixes = [f"{cfg.product}/{year}/{jday:03d}/"]

    tasks = [
        asyncio.create_task(gcs_list_names(session, cfg.bucket, pfx, sem, cfg.timeout_s))
        for pfx in prefixes
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    names: List[str] = []
    for res in results:
        if isinstance(res, Exception):
            continue
        names.extend(res)

    # Keep only GLM-L2-LCFA objects (prefix already narrows this, but safe)
    names = [n for n in names if cfg.product in n]
    return names


# ----------------------------
# Main
# ----------------------------

async def run(bucket: str, start: str, end: str, out_csv: str, list_conc: int):
    cfg = Cfg(bucket=bucket, list_conc=list_conc)
    sem = asyncio.Semaphore(cfg.list_conc)

    connector = aiohttp.TCPConnector(limit=cfg.list_conc, limit_per_host=cfg.list_conc, ttl_dns_cache=300)
    counts: Dict[Tuple[int, int], int] = {}

    async with aiohttp.ClientSession(connector=connector) as session:
        days = list(date_iter_utc(start, end))
        # Day-level concurrency too (don’t go wild)
        day_sem = asyncio.Semaphore(min(8, cfg.list_conc))

        async def one_day(day: datetime):
            async with day_sem:
                names = await list_day_glm(session, cfg, day, sem)
            # Count by parsing _sYYYYJJJ...
            c = 0
            for n in names:
                yj = parse_scan_start(n)
                if yj is None:
                    continue
                # Only count those whose parsed date matches the day we asked for
                # (guards against any odd listing overlap)
                if yj[0] == day.year and yj[1] == int(day.strftime("%j")):
                    c += 1
            counts[(day.year, int(day.strftime("%j")))] = c

        await asyncio.gather(*(one_day(d) for d in days))

    # Write CSV (sorted)
    rows = []
    for (year, jday), c in sorted(counts.items()):
        date = (datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=jday - 1)).date().isoformat()
        rows.append((date, year, jday, c))

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date_utc", "year", "jday", "glm_l2_lcfa_file_count"])
        w.writerows(rows)

    # Print summary
    total = sum(r[3] for r in rows)
    minc = min((r[3] for r in rows), default=0)
    maxc = max((r[3] for r in rows), default=0)
    print(f"\nBucket: {bucket}")
    print(f"Range:  {start} -> {end} (UTC)")
    print(f"Days:   {len(rows)}")
    print(f"Total GLM files: {total}")
    print(f"Per-day min/max: {minc} / {maxc}")
    print(f"Wrote: {out_csv}\n")


def main():
    ap = argparse.ArgumentParser(description="Count GLM-L2-LCFA files per day in a public GOES GCS bucket.")
    ap.add_argument("--bucket", default="gcp-public-data-goes-19", help="e.g. gcp-public-data-goes-16 / -18 / -19")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--out", default="glm_l2_lcfa_counts_per_day.csv", help="Output CSV path")
    ap.add_argument("--list-conc", type=int, default=64, help="Concurrent listing requests")
    args = ap.parse_args()

    asyncio.run(run(args.bucket, args.start, args.end, args.out, args.list_conc))


if __name__ == "__main__":
    main()