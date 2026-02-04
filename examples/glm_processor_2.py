import asyncio
import aiohttp
from google.cloud import storage
from datetime import datetime, timedelta
import pandas as pd
import sys
import os
import time
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor

# ----------------------------
# Config + Stats
# ----------------------------
@dataclass(frozen=True)
class Cfg:
    retries: int = 3
    timeout_s: int = 30

    dl_conc: int = 64
    proc_workers: int = 6

    list_q_max: int = 5000
    proc_q_max: int = 64 * 2
    out_q_max: int = 8 * 8

    csv_path: str = "filtered_data.csv"
    flush_frames: int = 500


@dataclass
class Stats:
    t0: float
    listed: int = 0
    list_errors: int = 0
    fatal_error: str | None = None

    downloaded: int = 0
    dl_errors: int = 0
    dl_bytes: int = 0

    proc_ok: int = 0
    proc_drop: int = 0
    proc_err: int = 0

    frames_in: int = 0
    frames_flushed: int = 0
    rows_written: int = 0
    written_bytes: int = 0


cfg = Cfg()
client = storage.Client.create_anonymous_client()

LIST_ERROR = "__LIST_ERROR__"


# ----------------------------
# Formatting helpers
# ----------------------------
def b2mb(x: float) -> float:
    return x / (1024.0 * 1024.0)


def fbytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f} {u}" if u != "B" else f"{int(x)} B"
        x /= 1024.0
    return f"{x:.2f} PB"


def fdur(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def getsize(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


# ----------------------------
# Date prefixes
# ----------------------------
def prefixes(start_date: str, end_date: str) -> list[str]:
    a = datetime.strptime(start_date, "%Y-%m-%d")
    b = datetime.strptime(end_date, "%Y-%m-%d")
    out = []
    while a <= b:
        out.append(f"GLM-L2-LCFA/{a.year}/{a.strftime('%j')}/")
        a += timedelta(days=1)
    return out


# ----------------------------
# Listing (simple + readable)
# ----------------------------
def _list_blob_names(bucket_name: str, prefix: str) -> list[str]:
    bucket = client.bucket(bucket_name)
    return [blob.name for blob in bucket.list_blobs(prefix=prefix)]


async def list_producer(bucket_name: str, pfxs: list[str], dl_q: asyncio.Queue, st: Stats):
    """
    Produces (seq, blob_name) into dl_q.
    On listing error: broadcasts LIST_ERROR to ALL downloaders so nothing hangs.
    """
    seq = 0
    try:
        for pfx in pfxs:
            names = await asyncio.to_thread(_list_blob_names, bucket_name, pfx)
            for name in names:
                await dl_q.put((seq, name))
                seq += 1
            st.listed += len(names)
    except Exception as e:
        st.list_errors += 1
        st.fatal_error = repr(e)
        for _ in range(cfg.dl_conc):
            await dl_q.put((LIST_ERROR, st.fatal_error))


# ----------------------------
# Download
# ----------------------------
async def fetch_bytes(session: aiohttp.ClientSession, url: str) -> bytes | None:
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
    print(f"\nDownload failed: {url} ({last})")
    return None


async def download_worker(bucket: str, dl_q: asyncio.Queue, proc_q: asyncio.Queue, session: aiohttp.ClientSession, st: Stats):
    while True:
        item = await dl_q.get()
        try:
            if item is None:
                return
            if isinstance(item, tuple) and item[0] == LIST_ERROR:
                print(f"\nListing error: {item[1]}")
                return

            seq, name = item
            url = f"https://storage.googleapis.com/{bucket}/{name}"
            payload = await fetch_bytes(session, url)

            if payload is None:
                st.dl_errors += 1
                await proc_q.put((seq, None))
            else:
                st.downloaded += 1
                st.dl_bytes += len(payload)
                await proc_q.put((seq, payload))
        finally:
            dl_q.task_done()


# ----------------------------
# NetCDF parsing/filtering (process)
# ----------------------------
def parse_and_filter_netcdf(payload: bytes, fields: list[str], latb: tuple[float, float], lonb: tuple[float, float], qmax: int) -> pd.DataFrame | None:
    import netCDF4 as nc
    import pandas as pd
    import numpy as np

    j2000 = pd.Timestamp("2000-01-01 12:00:00")

    try:
        with nc.Dataset("inmem", mode="r", memory=payload) as ds:
            if "flash_lat" not in ds.variables or "flash_lon" not in ds.variables:
                return None

            lat = np.asarray(ds.variables["flash_lat"][:])
            lon = np.asarray(ds.variables["flash_lon"][:])
            if lat.ndim != 1 or lon.ndim != 1:
                return None
            n = lat.shape[0]
            if n == 0:
                return None

            mask = (latb[0] <= lat) & (lat <= latb[1]) & (lonb[0] <= lon) & (lon <= lonb[1])

            if "flash_quality_flag" in ds.variables:
                q = np.asarray(ds.variables["flash_quality_flag"][:])
                if q.ndim != 1:
                    return None
                if q.size == 1 and n > 1:
                    q = np.full(n, q.item(), dtype=q.dtype)
                if q.shape[0] != n:
                    return None
                mask &= (q <= qmax)

            if not mask.any():
                return None

            def var_as_n(name: str):
                if name not in ds.variables:
                    return None
                a = np.asarray(ds.variables[name][:])
                if a.ndim == 0:
                    return np.full(n, a.item())
                if a.ndim == 1 and a.size == 1 and n > 1:
                    return np.full(n, a.item(), dtype=a.dtype)
                if a.ndim == 1 and a.shape[0] == n:
                    return a
                return None

            out = {}

            # Always read product_time internally to compute product_time_offset,
            # but do NOT include product_time in output.
            pt = var_as_n("product_time")
            if pt is not None:
                pt = pt[mask]
                out["product_time_offset"] = j2000 + pd.to_timedelta(pt, unit="s")

            # Add only requested fields (flash_id removed; flash_time_offset removed; product_time removed)
            for f in fields:
                a = var_as_n(f)
                if a is not None:
                    out[f] = a[mask]

            if not out:
                return None

            df = pd.DataFrame(out)
            if df.empty:
                return None

            # Sort by product_time_offset (since product_time is not present)
            if "product_time_offset" in df.columns:
                df = df.sort_values("product_time_offset", kind="mergesort").reset_index(drop=True)

            return df
    except Exception:
        return None


async def process_worker(proc_q: asyncio.Queue, out_q: asyncio.Queue, pool: ProcessPoolExecutor, fields, latb, lonb, qmax, st: Stats):
    loop = asyncio.get_running_loop()
    while True:
        item = await proc_q.get()
        try:
            if item is None:
                return
            seq, payload = item
            if payload is None:
                st.proc_drop += 1
                await out_q.put((seq, None))
                continue

            df = await loop.run_in_executor(pool, parse_and_filter_netcdf, payload, fields, latb, lonb, qmax)
            if df is None:
                st.proc_drop += 1
                await out_q.put((seq, None))
            else:
                st.proc_ok += 1
                await out_q.put((seq, df))
        except Exception:
            st.proc_err += 1
            if isinstance(item, tuple) and isinstance(item[0], int):
                await out_q.put((item[0], None))
        finally:
            proc_q.task_done()


# ----------------------------
# Ordered writer (frames only)
# ----------------------------
async def ordered_csv_writer(out_q: asyncio.Queue, st: Stats, path: str):
    pending: dict[int, pd.DataFrame | None] = {}
    next_seq = 0
    buf: list[pd.DataFrame] = []
    header_written = os.path.exists(path) and getsize(path) > 0

    def flush_sync():
        nonlocal buf, header_written
        if not buf:
            return
        before = getsize(path)
        df_out = pd.concat(buf, ignore_index=True)
        df_out.to_csv(path, mode="a", index=False, header=not header_written)
        header_written = True
        after = getsize(path)

        st.written_bytes += max(0, after - before)
        st.frames_flushed += len(buf)
        st.rows_written += len(df_out)
        buf = []

    async def flush():
        await asyncio.to_thread(flush_sync)

    def maybe_add(df: pd.DataFrame | None):
        if df is None or df.empty:
            return
        buf.append(df)
        st.frames_in += 1

    while True:
        item = await out_q.get()
        try:
            if item is None:
                while next_seq in pending:
                    maybe_add(pending.pop(next_seq))
                    next_seq += 1
                    if len(buf) >= cfg.flush_frames:
                        await flush()
                await flush()
                return

            seq, df = item
            pending[seq] = df

            while next_seq in pending:
                maybe_add(pending.pop(next_seq))
                next_seq += 1
                if len(buf) >= cfg.flush_frames:
                    await flush()
        finally:
            out_q.task_done()


# ----------------------------
# Progress
# ----------------------------
async def progress(st: Stats, path: str):
    tty = sys.stdout.isatty()
    started = False
    while True:
        t = time.perf_counter() - st.t0
        t_safe = max(t, 1e-6)
        out_size = getsize(path)

        line1 = f"t={fdur(t)}  dl={fbytes(st.dl_bytes)} ({b2mb(st.dl_bytes)/t_safe:.2f} MB/s)  out={fbytes(out_size)} ({b2mb(out_size)/t_safe:.2f} MB/s)"
        line2 = f"listed={st.listed}  downloaded={st.downloaded}  dl_err={st.dl_errors}  proc_ok={st.proc_ok}  proc_drop={st.proc_drop}  proc_err={st.proc_err}"
        line3 = f"frames_in={st.frames_in}  frames_flushed={st.frames_flushed}  rows_written={st.rows_written}"

        if tty:
            if started:
                sys.stdout.write("\x1b[3A")
            else:
                started = True
            sys.stdout.write("\x1b[2K" + line1 + "\n")
            sys.stdout.write("\x1b[2K" + line2 + "\n")
            sys.stdout.write("\x1b[2K" + line3 + "\n")
        else:
            sys.stdout.write("\r" + line1 + " | " + line2 + " | " + line3)

        sys.stdout.flush()
        await asyncio.sleep(0.5)


# ----------------------------
# Main pipeline
# ----------------------------
async def main_async(bucket: str, pfxs: list[str], fields: list[str], latb, lonb, qmax: int = 1, out_path: str = cfg.csv_path) -> str:
    dl_q = asyncio.Queue(maxsize=cfg.list_q_max)
    proc_q = asyncio.Queue(maxsize=cfg.proc_q_max)
    out_q = asyncio.Queue(maxsize=cfg.out_q_max)

    if os.path.exists(out_path):
        os.remove(out_path)

    st = Stats(t0=time.perf_counter())
    prog_task = asyncio.create_task(progress(st, out_path))
    writer_task = asyncio.create_task(ordered_csv_writer(out_q, st, out_path))

    connector = aiohttp.TCPConnector(limit=cfg.dl_conc, limit_per_host=cfg.dl_conc, ttl_dns_cache=300, enable_cleanup_closed=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        prod_task = asyncio.create_task(list_producer(bucket, pfxs, dl_q, st))
        dl_tasks = [asyncio.create_task(download_worker(bucket, dl_q, proc_q, session, st)) for _ in range(cfg.dl_conc)]

        with ProcessPoolExecutor(max_workers=cfg.proc_workers) as pool:
            proc_tasks = [
                asyncio.create_task(process_worker(proc_q, out_q, pool, fields, latb, lonb, qmax, st))
                for _ in range(cfg.proc_workers)
            ]

            await prod_task

            # stop downloaders unless listing error already told them to exit
            if st.fatal_error is None:
                for _ in range(cfg.dl_conc):
                    await dl_q.put(None)

            await dl_q.join()
            await asyncio.gather(*dl_tasks, return_exceptions=True)

            # stop processors
            for _ in proc_tasks:
                await proc_q.put(None)

            await proc_q.join()
            await asyncio.gather(*proc_tasks, return_exceptions=True)

    # stop writer
    await out_q.put(None)
    await out_q.join()
    await writer_task

    # stop progress
    prog_task.cancel()
    try:
        await prog_task
    except asyncio.CancelledError:
        pass

    # final summary
    elapsed = time.perf_counter() - st.t0
    out_size = getsize(out_path)

    print("\n" + "-" * 72)
    print("FINAL")
    print("-" * 72)
    print(f"Elapsed:                   {fdur(elapsed)} ({elapsed:.2f}s)")
    print(f"Listed:                    {st.listed}")
    if st.fatal_error:
        print(f"Listing fatal error:       {st.fatal_error}")
    print(f"Downloaded:                {st.downloaded} (errors: {st.dl_errors})")
    print(f"Processed ok/drop/err:     {st.proc_ok}/{st.proc_drop}/{st.proc_err}")
    print(f"Download bytes:            {fbytes(st.dl_bytes)} ({b2mb(st.dl_bytes)/max(elapsed,1e-6):.2f} MB/s)")
    print(f"Output size:               {fbytes(out_size)} ({b2mb(out_size)/max(elapsed,1e-6):.2f} MB/s)")
    print(f"Written-by-flush delta:    {fbytes(st.written_bytes)}")
    print(f"Frames in/flushed:         {st.frames_in}/{st.frames_flushed}")
    print(f"Rows written:              {st.rows_written}")
    print("-" * 72 + "\n")

    return out_path


def main():
    bucket = "gcp-public-data-goes-16"
    start_date, end_date = "2024-01-01", "2025-01-01"

    # Removed: flash_id, flash_time_offset_of_first_event, product_time
    # Kept: product_time_offset (computed internally), plus the other fields below.
    fields = [
        "flash_lat",
        "flash_lon",
        "flash_quality_flag",
        "flash_energy",
    ]

    latb = (0.0, 55.0)
    lonb = (-135.0, -45.0)

    out_path = asyncio.run(main_async(bucket, prefixes(start_date, end_date), fields, latb, lonb, qmax=1, out_path=cfg.csv_path))

    if os.path.exists(out_path) and getsize(out_path) > 0:
        print(f"Data saved to {out_path}\n")
    else:
        print("No data written (or all rows filtered out).\n")


if __name__ == "__main__":
    main()
