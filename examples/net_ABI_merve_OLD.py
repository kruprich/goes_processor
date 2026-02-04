"""
Multi-day concurrent, in-memory ABI L2 extraction + merge (writes ONLY merged Zarr per day)

CHUNKED per product/day:
- For each product/day, process files in chunks of size cfg.frames_in_flight.
- Within a chunk, downloads overlap (bounded by http_sem) and decodes run in process pool.
- Keeps memory stable without serializing the whole day and without spawning thousands of tasks.

What it does (per UTC day):
  1) List ABI L2 files in GCS for:
       - CMIP CONUS C13: ABI-L2-CMIPC-M6C13
       - ACHA CONUS:     ABI-L2-ACHAC-M6
       - TPW  CONUS:     ABI-L2-TPWC-M6
  2) Download each NetCDF into memory (bytes), decode value + DQF, keep ONLY dqf==0 pixels
  3) Coarsen native 2 km grid -> 8 km grid via block mean over valid pixels + valid fraction
  4) For exact 5-min bins (00:00..23:55 UTC), pick nearest scan per product within tolerance
  5) Stack channels and write merged daily Zarr:
       X[time, channel, y, x] channels:
         0 cmip_c13_cmi
         1 acha_ht
         2 tpw_tpw
         3 cmip_validfrac
         4 acha_validfrac
         5 tpw_validfrac

No intermediate per-product Zarrs. No NetCDF saved to disk.

Deps:
  pip install google-cloud-storage aiohttp netCDF4 numpy xarray zarr

Run:
  python abi_l2_mem_merge_multiday_chunked.py
"""

from __future__ import annotations

import os
import re
import sys
import json
import shutil
import asyncio
import aiohttp
import numpy as np
import xarray as xr

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from concurrent.futures import ProcessPoolExecutor
from google.cloud import storage


# =============================================================================
# Config
# =============================================================================

@dataclass(frozen=True)
class ProductCfg:
    product: str
    must_contain: str
    value_var: str
    dqf_var: str
    out_name: str  # label only


@dataclass(frozen=True)
class Cfg:
    bucket: str = "gcp-public-data-goes-16"

    # UTC date range
    start_date: str = "2023-01-01"
    end_date: str = "2023-01-31"

    # Native 2 km -> 8 km coarsen
    coarsen_factor: int = 4

    # Nearest-time tolerance for 5-min bin merge
    tol_s: int = 180  # ±3 minutes

    # Concurrency (global HTTP)
    dl_conc: int = 128
    timeout_s: int = 60
    retries: int = 3

    # Decode/CPU
    proc_workers: int = 12

    # Concurrency (across days)
    days_in_flight: int = 1  # your earlier note: start 2; try 3 if RAM stable

    # Concurrency (within a product/day) — CHUNK SIZE
    frames_in_flight: int = 16  # try 8..32

    # Output (writes ONLY merged daily zarr)
    out_root: str = "./abi_l2_conus_v1_memonly"
    overwrite: bool = False

    # Merge behavior
    drop_if_missing: bool = True  # if any product missing/outside tolerance, drop the 5-min bin


cfg = Cfg()

PROD_CMIP = ProductCfg(
    product="ABI-L2-CMIPC",
    must_contain="ABI-L2-CMIPC-M6C13",
    value_var="CMI",
    dqf_var="DQF",
    out_name="cmip_c13",
)

PROD_ACHA = ProductCfg(
    product="ABI-L2-ACHAC",
    must_contain="ABI-L2-ACHAC-M6",
    value_var="HT",
    dqf_var="DQF",
    out_name="acha",
)

PROD_TPW = ProductCfg(
    product="ABI-L2-TPWC",
    must_contain="ABI-L2-TPWC-M6",
    value_var="TPW",
    dqf_var="DQF_Overall",
    out_name="tpw",
)

PRODUCTS = [PROD_CMIP, PROD_ACHA, PROD_TPW]


# =============================================================================
# Globals / regex / client
# =============================================================================

client = storage.Client.create_anonymous_client()
_PAT = re.compile(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})")


# =============================================================================
# Utilities
# =============================================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def rm_tree(path: str):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)

def zarr_exists(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, ".zgroup"))

def parse_scan_start_utc(blob_name: str) -> datetime | None:
    m = _PAT.search(blob_name)
    if not m:
        return None
    y, j, hh, mm, ss = map(int, m.groups())
    return datetime(y, 1, 1, tzinfo=timezone.utc) + timedelta(days=j - 1, hours=hh, minutes=mm, seconds=ss)

def yjjj(dt: datetime) -> tuple[int, int]:
    return dt.year, int(dt.strftime("%j"))

def date_iter_utc(start_date: str, end_date: str):
    a = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    b = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    cur = a
    while cur <= b:
        yield cur
        cur += timedelta(days=1)

def build_5min_bins_for_day(year: int, jday: int) -> np.ndarray:
    day = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=jday - 1)
    start = np.datetime64(day.replace(tzinfo=None), "ns")
    bins = start + (np.arange(0, 24 * 60, 5, dtype=np.int64) * np.timedelta64(1, "m"))
    return bins  # 288 bins

def merged_path(out_root: str, year: int, jday: int) -> str:
    d = os.path.join(out_root, "merged", str(year))
    ensure_dir(d)
    return os.path.join(d, f"{year}{jday:03d}.zarr")

def print_cfg():
    print("-" * 72)
    print("CONFIG")
    print("-" * 72)
    print(json.dumps(cfg.__dict__, indent=2))
    print("-" * 72)


# =============================================================================
# GCS listing
# =============================================================================

def list_hour(product: str, year: int, jday: int, hour: int) -> list[str]:
    prefix = f"{product}/{year}/{jday:03d}/{hour:02d}/"
    b = client.bucket(cfg.bucket)
    return [obj.name for obj in b.list_blobs(prefix=prefix)]

def list_day_product(prod: ProductCfg, year: int, jday: int) -> list[tuple[datetime, str]]:
    names: list[str] = []
    for hh in range(24):
        names.extend(list_hour(prod.product, year, jday, hh))

    pairs: list[tuple[datetime, str]] = []
    for n in names:
        if prod.must_contain not in n:
            continue
        t = parse_scan_start_utc(n)
        if t is not None:
            pairs.append((t, n))
    pairs.sort(key=lambda x: x[0])
    return pairs


# =============================================================================
# Download
# =============================================================================

async def fetch_bytes(session: aiohttp.ClientSession, blob_name: str) -> bytes | None:
    url = f"https://storage.googleapis.com/{cfg.bucket}/{blob_name}"
    timeout = aiohttp.ClientTimeout(total=cfg.timeout_s)
    last = None
    for attempt in range(1, cfg.retries + 1):
        try:
            async with session.get(url, timeout=timeout) as r:
                if r.status == 200:
                    return await r.read()
                last = f"HTTP {r.status}"
        except Exception as e:
            last = repr(e)
        await asyncio.sleep(min(0.25 * attempt, 1.0))
    print(f"  !! download failed ({last}): {blob_name}")
    return None


# =============================================================================
# NetCDF decode + coarsen (DQF==0)
# =============================================================================

def extract_mean_valid_and_frac(payload: bytes, value_var: str, dqf_var: str, factor: int):
    """
    Returns:
      mean_over_valid[y8,x8], valid_fraction[y8,x8] as float32
    """
    import netCDF4 as nc
    import numpy as np

    def trim(a):
        y, x = a.shape
        y2 = (y // factor) * factor
        x2 = (x // factor) * factor
        return a[:y2, :x2]

    def block_sum(a):
        a = trim(a)
        y, x = a.shape
        return a.reshape(y // factor, factor, x // factor, factor).sum(axis=(1, 3))

    def block_frac(valid):
        v = trim(valid).astype(np.float32)
        y, x = v.shape
        return v.reshape(y // factor, factor, x // factor, factor).mean(axis=(1, 3))

    with nc.Dataset("inmem", mode="r", memory=payload) as ds:
        val = np.asarray(ds.variables[value_var][:], dtype=np.float32)
        dqf = np.asarray(ds.variables[dqf_var][:], dtype=np.int16)

        valid = (dqf == 0)
        val_sum = block_sum(np.where(valid, val, 0.0).astype(np.float32))
        frac = block_frac(valid)  # 0..1
        cnt = np.maximum(frac * (factor * factor), 1.0)  # avoid div0
        mean_valid = val_sum / cnt
        return mean_valid.astype(np.float32), frac.astype(np.float32)

def _decode_one(payload: bytes, value_var: str, dqf_var: str, factor: int):
    return extract_mean_valid_and_frac(payload, value_var, dqf_var, factor)


# =============================================================================
# Stage A (in-memory): extract product-day -> (times, value, validfrac)
# CHUNKED: process day files in chunks of cfg.frames_in_flight
# =============================================================================

async def extract_product_day_inmem(
    prod: ProductCfg,
    year: int,
    jday: int,
    session: aiohttp.ClientSession,
    http_sem: asyncio.Semaphore,
    pool: ProcessPoolExecutor
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:

    pairs = list_day_product(prod, year, jday)
    if not pairs:
        print(f"  !! no files for {prod.out_name} on {year}{jday:03d}")
        return None

    loop = asyncio.get_running_loop()
    out: list[tuple[np.datetime64, np.ndarray, np.ndarray]] = []

    # This bounds tasks AND bounds payloads-in-flight per product/day.
    chunk_n = max(1, int(cfg.frames_in_flight))

    async def one(t: datetime, blob: str):
        # Download is bounded globally by http_sem
        async with http_sem:
            payload = await fetch_bytes(session, blob)
        if payload is None:
            return None

        # Decode in process pool (CPU-bound)
        v, f = await loop.run_in_executor(
            pool, _decode_one, payload, prod.value_var, prod.dqf_var, cfg.coarsen_factor
        )

        t64 = np.datetime64(t.replace(tzinfo=None), "ns")
        return (t64, v, f)

    # Process in chunks to avoid spawning many tasks at once.
    for i in range(0, len(pairs), chunk_n):
        chunk = pairs[i : i + chunk_n]
        tasks = [asyncio.create_task(one(t, blob)) for (t, blob) in chunk]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if r is None:
                continue
            if isinstance(r, Exception):
                # keep going; you can print here if you want
                continue
            out.append(r)

    if not out:
        print(f"  !! decode produced 0 frames for {prod.out_name} on {year}{jday:03d}")
        return None

    out.sort(key=lambda x: x[0])
    times = np.array([r[0] for r in out], dtype="datetime64[ns]")
    values = np.stack([r[1] for r in out], axis=0).astype(np.float32)  # (time,y,x)
    valids = np.stack([r[2] for r in out], axis=0).astype(np.float32)  # (time,y,x)
    return times, values, valids


# =============================================================================
# Stage B: merge in-memory extracts to 5-min bins
# =============================================================================

def nearest_index(times: np.ndarray, t0: np.datetime64) -> int | None:
    if times.size == 0:
        return None
    i = np.searchsorted(times, t0)
    cand = []
    if i < len(times):
        cand.append(i)
    if i > 0:
        cand.append(i - 1)
    if not cand:
        return None
    best = min(cand, key=lambda j: abs((times[j] - t0) / np.timedelta64(1, "s")))
    return int(best)

def merge_day_inmem(
    *,
    year: int,
    jday: int,
    cm_times: np.ndarray, cm_val: np.ndarray, cm_vf: np.ndarray,
    ac_times: np.ndarray, ac_val: np.ndarray, ac_vf: np.ndarray,
    tp_times: np.ndarray, tp_val: np.ndarray, tp_vf: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    bins = build_5min_bins_for_day(year, jday)
    tol = float(cfg.tol_s)

    # Crop ONCE per day to a common shape (fixes shape mismatch across products)
    y0 = min(cm_val.shape[1], ac_val.shape[1], tp_val.shape[1])
    x0 = min(cm_val.shape[2], ac_val.shape[2], tp_val.shape[2])

    def crop(a):
        return a[..., :y0, :x0]

    cm_val = crop(cm_val); cm_vf = crop(cm_vf)
    ac_val = crop(ac_val); ac_vf = crop(ac_vf)
    tp_val = crop(tp_val); tp_vf = crop(tp_vf)

    X_list: list[np.ndarray] = []
    t_list: list[np.datetime64] = []

    for t0 in bins:
        i_cm = nearest_index(cm_times, t0)
        i_ac = nearest_index(ac_times, t0)
        i_tp = nearest_index(tp_times, t0)
        if i_cm is None or i_ac is None or i_tp is None:
            if cfg.drop_if_missing:
                continue
            else:
                continue

        dt_cm = abs((cm_times[i_cm] - t0) / np.timedelta64(1, "s"))
        dt_ac = abs((ac_times[i_ac] - t0) / np.timedelta64(1, "s"))
        dt_tp = abs((tp_times[i_tp] - t0) / np.timedelta64(1, "s"))

        if dt_cm > tol or dt_ac > tol or dt_tp > tol:
            if cfg.drop_if_missing:
                continue
            else:
                continue

        X = np.stack(
            [
                cm_val[i_cm], ac_val[i_ac], tp_val[i_tp],
                cm_vf[i_cm],  ac_vf[i_ac],  tp_vf[i_tp],
            ],
            axis=0
        ).astype(np.float32)  # (6,y,x)

        X_list.append(X)
        t_list.append(t0)

    if not X_list:
        return None

    X_all = np.stack(X_list, axis=0).astype(np.float32)  # (time,6,y,x)
    t_used = np.array(t_list, dtype="datetime64[ns]")
    return t_used, X_all


# =============================================================================
# Write merged day (ONLY persistent artifact)
# =============================================================================

def write_merged_day(out_zarr: str, t_used: np.ndarray, X_all: np.ndarray, year: int, jday: int):
    if zarr_exists(out_zarr) and not cfg.overwrite:
        return
    if cfg.overwrite:
        rm_tree(out_zarr)

    ds = xr.Dataset(
        {"X": (("time", "channel", "y", "x"), X_all)},
        coords={
            "time": t_used,
            "channel": np.array(
                ["cmip_c13_cmi", "acha_ht", "tpw_tpw", "cmip_validfrac", "acha_validfrac", "tpw_validfrac"]
            ),
            "y": np.arange(X_all.shape[2]),
            "x": np.arange(X_all.shape[3]),
        },
        attrs={
            "year": year,
            "jday": jday,
            "tol_s": cfg.tol_s,
            "coarsen_factor": cfg.coarsen_factor,
            "drop_if_missing": cfg.drop_if_missing,
        }
    )

    enc = {"X": {"chunks": (1, 6, 256, 256)}}
    ds.to_zarr(out_zarr, mode="w", encoding=enc)


# =============================================================================
# Per-day orchestration
# =============================================================================

async def process_one_day(day: datetime, session, http_sem, pool) -> bool:
    year, jday = yjjj(day)
    out_zarr = merged_path(cfg.out_root, year, jday)

    if zarr_exists(out_zarr) and not cfg.overwrite:
        print(f"SKIP {day.date().isoformat()} ({year}{jday:03d}) already merged")
        return True

    print(f"\n=== {day.date().isoformat()} ({year}{jday:03d}) ===")

    cm = await extract_product_day_inmem(PROD_CMIP, year, jday, session, http_sem, pool)
    ac = await extract_product_day_inmem(PROD_ACHA, year, jday, session, http_sem, pool)
    tp = await extract_product_day_inmem(PROD_TPW,  year, jday, session, http_sem, pool)

    if cm is None or ac is None or tp is None:
        print("  !! extraction failed for at least one product; skipping day")
        cm = ac = tp = None
        return False

    cm_times, cm_val, cm_vf = cm
    ac_times, ac_val, ac_vf = ac
    tp_times, tp_val, tp_vf = tp

    merged = merge_day_inmem(
        year=year, jday=jday,
        cm_times=cm_times, cm_val=cm_val, cm_vf=cm_vf,
        ac_times=ac_times, ac_val=ac_val, ac_vf=ac_vf,
        tp_times=tp_times, tp_val=tp_val, tp_vf=tp_vf,
    )

    if merged is None:
        print("  !! merge produced 0 bins (try increasing tol_s)")
        del cm_times, cm_val, cm_vf, ac_times, ac_val, ac_vf, tp_times, tp_val, tp_vf
        return False

    t_used, X_all = merged
    write_merged_day(out_zarr, t_used, X_all, year, jday)
    print(f"  -> wrote merged: {out_zarr}  bins={len(t_used)}  grid={X_all.shape[2]}x{X_all.shape[3]}")

    # Free memory aggressively (important when days_in_flight > 1)
    del cm_times, cm_val, cm_vf, ac_times, ac_val, ac_vf, tp_times, tp_val, tp_vf, t_used, X_all
    return True


# =============================================================================
# Multi-day runner (concurrent days)
# =============================================================================

async def run_all():
    ensure_dir(cfg.out_root)
    print_cfg()

    days = list(date_iter_utc(cfg.start_date, cfg.end_date))
    total = len(days)

    # Shared HTTP session + shared process pool
    connector = aiohttp.TCPConnector(limit=cfg.dl_conc, limit_per_host=cfg.dl_conc, ttl_dns_cache=300)
    http_sem = asyncio.Semaphore(cfg.dl_conc)

    q: asyncio.Queue[datetime | None] = asyncio.Queue()
    for d in days:
        await q.put(d)
    for _ in range(cfg.days_in_flight):
        await q.put(None)

    ok = 0
    fail = 0
    counter_lock = asyncio.Lock()

    async with aiohttp.ClientSession(connector=connector) as session:
        with ProcessPoolExecutor(max_workers=cfg.proc_workers) as pool:

            async def worker(wid: int):
                nonlocal ok, fail
                while True:
                    day = await q.get()
                    try:
                        if day is None:
                            return
                        good = await process_one_day(day, session, http_sem, pool)
                        async with counter_lock:
                            if good:
                                ok += 1
                            else:
                                fail += 1
                    finally:
                        q.task_done()

            tasks = [asyncio.create_task(worker(i)) for i in range(cfg.days_in_flight)]
            await q.join()
            await asyncio.gather(*tasks, return_exceptions=True)

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"Days attempted: {total}")
    print(f"Days merged OK: {ok}")
    print(f"Days failed:    {fail}")
    print(f"Output root:    {os.path.abspath(cfg.out_root)}")
    print("=" * 72)


# =============================================================================
# Main
# =============================================================================

def main():
    try:
        import netCDF4  # noqa
        import zarr     # noqa
    except Exception:
        print("Missing deps. Install:")
        print("  pip install aiohttp google-cloud-storage netCDF4 numpy xarray zarr")
        sys.exit(1)

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
