import asyncio
import aiohttp
from google.cloud import storage
from datetime import datetime, timedelta
import pandas as pd
import sys
import contextlib
import os
from concurrent.futures import ProcessPoolExecutor

# ----------------------------
# Config
# ----------------------------
RETRIES = 3
MAX_DOWNLOAD_CONCURRENCY = 64

# Bounded listing queue so blob-name buffer can't grow without bound
LIST_QUEUE_MAXSIZE = 5000  # tune: 2k–20k

# For macOS + netCDF4, use processes (NOT threads) for parsing
MAX_PROCESS_WORKERS = 8

TIMEOUT_SECONDS = 30
PROCESS_QUEUE_MAXSIZE = MAX_DOWNLOAD_CONCURRENCY * 2

# CSV flushing config (frames-only; NO time-based flush)
OUTPUT_CSV_PATH = "filtered_data.csv"
CSV_FLUSH_EVERY_FRAMES = 500

# J2000 epoch used by GOES-R time fields (seconds since 2000-01-01 12:00:00 UTC)
J2000_EPOCH = pd.Timestamp("2000-01-01 12:00:00")  # UTC-naive timestamp for CSV

client = storage.Client.create_anonymous_client()


# ----------------------------
# Date prefixes
# ----------------------------
def generate_prefixes(start_date: str, end_date: str) -> list[str]:
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    prefixes = []
    while start_dt <= end_dt:
        year = start_dt.year
        julian_day = start_dt.strftime("%j")
        prefixes.append(f"GLM-L2-LCFA/{year}/{julian_day}/")
        start_dt += timedelta(days=1)
    return prefixes


# ----------------------------
# Download
# ----------------------------
async def download_blob_bytes(session: aiohttp.ClientSession, url: str, retries: int = RETRIES) -> bytes | None:
    timeout = aiohttp.ClientTimeout(total=TIMEOUT_SECONDS)
    last_err = None

    for attempt in range(1, retries + 1):
        try:
            async with session.get(url, timeout=timeout) as resp:
                if resp.status == 200:
                    return await resp.read()
                last_err = f"HTTP {resp.status}"
        except Exception as e:
            last_err = repr(e)

        await asyncio.sleep(min(0.25 * attempt, 1.0))

    print(f"Download failed: {url} ({last_err})")
    return None


async def list_blobs_producer(bucket_name: str, prefixes: list[str], download_queue: asyncio.Queue, counters: dict):
    """
    Lists blob names and enqueues them WITH a monotonically increasing seq.
    The queue is bounded, so listing backpressures naturally.
    """
    loop = asyncio.get_running_loop()

    async def _enqueue(seq: int, name: str):
        await download_queue.put((seq, name))
        counters["listed"] += 1

    def _thread_list():
        try:
            bucket = client.bucket(bucket_name)
            seq = 0
            for prefix in prefixes:
                for blob in bucket.list_blobs(prefix=prefix):
                    asyncio.run_coroutine_threadsafe(_enqueue(seq, blob.name), loop).result()
                    seq += 1
        except Exception as e:
            asyncio.run_coroutine_threadsafe(download_queue.put(("__LIST_ERROR__", repr(e))), loop).result()

    await asyncio.to_thread(_thread_list)


async def download_worker(bucket_name: str, download_queue: asyncio.Queue, process_queue: asyncio.Queue,
                          session: aiohttp.ClientSession, counters: dict):
    while True:
        item = await download_queue.get()
        try:
            if item is None:
                return

            if isinstance(item, tuple) and item and item[0] == "__LIST_ERROR__":
                counters["list_errors"] += 1
                print(f"\nListing error: {item[1]}")
                return

            seq, blob_name = item
            url = f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
            payload = await download_blob_bytes(session, url)

            if payload is None:
                counters["download_errors"] += 1
                # still emit a placeholder so ordered writer can advance
                await process_queue.put((seq, None))
            else:
                await process_queue.put((seq, payload))
                counters["downloaded"] += 1
        finally:
            download_queue.task_done()


# ----------------------------
# NetCDF parsing + filtering (RUNS IN A SEPARATE PROCESS)
# ----------------------------
def parse_and_filter_netcdf(payload: bytes,
                            fields: list[str],
                            lat_bounds: tuple[float, float],
                            lon_bounds: tuple[float, float],
                            quality_max: int) -> pd.DataFrame | None:
    import netCDF4 as nc
    import pandas as pd

    try:
        with nc.Dataset("inmem", mode="r", memory=payload) as ds:
            data = {}
            lengths = []

            for field in fields:
                if field in ds.variables:
                    arr = ds.variables[field][:]
                    data[field] = arr
                    try:
                        lengths.append(len(arr))
                    except TypeError:
                        lengths.append(None)

            if not data:
                return None

            clean_lengths = [l for l in lengths if l is not None]
            if clean_lengths and len(set(clean_lengths)) != 1:
                return None

            df = pd.DataFrame(data)
            if df.empty:
                return None

            # Add product_time_offset = J2000 epoch + product_time seconds
            if "product_time" in df.columns:
                df["product_time_offset"] = J2000_EPOCH + pd.to_timedelta(df["product_time"], unit="s")

            if "flash_lat" not in df.columns or "flash_lon" not in df.columns:
                return None

            mask = (
                (df["flash_lat"] >= lat_bounds[0]) & (df["flash_lat"] <= lat_bounds[1]) &
                (df["flash_lon"] >= lon_bounds[0]) & (df["flash_lon"] <= lon_bounds[1])
            )

            if "flash_quality_flag" in df.columns:
                mask &= (df["flash_quality_flag"] <= quality_max)

            filtered = df.loc[mask]
            if filtered.empty:
                return None

            # Per-file sort by product_time ONLY
            if "product_time" in filtered.columns:
                filtered = filtered.sort_values(["product_time"], kind="mergesort").reset_index(drop=True)

            return filtered

    except Exception:
        return None


async def process_consumer(process_queue: asyncio.Queue,
                           out_queue: asyncio.Queue,
                           pool: ProcessPoolExecutor,
                           fields: list[str],
                           lat_bounds: tuple[float, float],
                           lon_bounds: tuple[float, float],
                           quality_max: int,
                           counters: dict):
    """
    Consumes (seq, payload) and emits (seq, df_or_none) for EVERY seq.
    Required so ordered writer can always advance.
    """
    loop = asyncio.get_running_loop()

    while True:
        item = await process_queue.get()
        try:
            if item is None:
                return

            seq, payload = item

            if payload is None:
                counters["processed_empty_or_filtered"] += 1
                await out_queue.put((seq, None))
                continue

            df = await loop.run_in_executor(
                pool,
                parse_and_filter_netcdf,
                payload,
                fields,
                lat_bounds,
                lon_bounds,
                quality_max,
            )

            if df is None:
                counters["processed_empty_or_filtered"] += 1
                await out_queue.put((seq, None))
            else:
                counters["processed_ok"] += 1
                await out_queue.put((seq, df))

        except Exception as e:
            counters["process_errors"] += 1
            print(f"\nProcess error: {e}")
            # Emit drop to avoid stalling ordering
            try:
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], int):
                    await out_queue.put((item[0], None))
            except Exception:
                pass
        finally:
            process_queue.task_done()


# ----------------------------
# ORDERED CSV writer (frames-only flush; NO time-based flush)
# ----------------------------
async def csv_writer_ordered(
    out_queue: asyncio.Queue,
    counters: dict,
    output_csv_path: str,
    flush_every_frames: int = CSV_FLUSH_EVERY_FRAMES,
) -> None:
    """
    Ordered writer:
      - buffers out-of-order (seq, df_or_none) results
      - writes ONLY when next_seq is available
      - guarantees the CSV is appended in file order (seq order)
    """
    pending: dict[int, pd.DataFrame | None] = {}
    next_seq = 0

    buffer: list[pd.DataFrame] = []
    header_written = os.path.exists(output_csv_path) and os.path.getsize(output_csv_path) > 0

    def _flush_sync():
        nonlocal buffer, header_written
        if not buffer:
            return
        df_out = pd.concat(buffer, ignore_index=True)
        df_out.to_csv(output_csv_path, mode="a", index=False, header=not header_written)
        header_written = True
        counters["frames_flushed"] += len(buffer)
        counters["rows_written"] += len(df_out)
        buffer = []

    def _maybe_buffer(df: pd.DataFrame | None):
        if df is None or df.empty:
            return
        buffer.append(df)
        counters["frames_collected"] += 1
        if len(buffer) >= flush_every_frames:
            _flush_sync()

    while True:
        item = await out_queue.get()
        try:
            if item is None:
                while next_seq in pending:
                    _maybe_buffer(pending.pop(next_seq))
                    next_seq += 1
                _flush_sync()
                return

            seq, df = item
            pending[seq] = df

            while next_seq in pending:
                _maybe_buffer(pending.pop(next_seq))
                next_seq += 1

        finally:
            out_queue.task_done()


async def print_progress(counters: dict):
    while True:
        sys.stdout.write(
            "\r"
            f"listed={counters['listed']}  "
            f"downloaded={counters['downloaded']}  "
            f"dl_err={counters['download_errors']}  "
            f"proc_ok={counters['processed_ok']}  "
            f"proc_drop={counters['processed_empty_or_filtered']}  "
            f"proc_err={counters['process_errors']}  "
            f"frames_in={counters['frames_collected']}  "
            f"frames_flushed={counters['frames_flushed']}  "
            f"rows_written={counters['rows_written']}  "
        )
        sys.stdout.flush()
        await asyncio.sleep(0.5)


# ----------------------------
# Pipeline
# ----------------------------
async def main_async(bucket_name: str,
                     prefixes: list[str],
                     fields: list[str],
                     lat_bounds: tuple[float, float],
                     lon_bounds: tuple[float, float],
                     quality_max: int = 1,
                     output_csv_path: str = OUTPUT_CSV_PATH) -> str:

    download_queue = asyncio.Queue(maxsize=LIST_QUEUE_MAXSIZE)
    process_queue = asyncio.Queue(maxsize=PROCESS_QUEUE_MAXSIZE)
    out_queue = asyncio.Queue()

    if os.path.exists(output_csv_path):
        os.remove(output_csv_path)

    counters = {
        "listed": 0,
        "list_errors": 0,
        "downloaded": 0,
        "download_errors": 0,
        "processed_ok": 0,
        "processed_empty_or_filtered": 0,
        "process_errors": 0,
        "frames_collected": 0,
        "frames_flushed": 0,
        "rows_written": 0,
    }

    progress_task = asyncio.create_task(print_progress(counters))
    writer_task = asyncio.create_task(csv_writer_ordered(out_queue, counters, output_csv_path))

    connector = aiohttp.TCPConnector(limit=MAX_DOWNLOAD_CONCURRENCY)
    async with aiohttp.ClientSession(connector=connector) as session:
        producer_task = asyncio.create_task(list_blobs_producer(bucket_name, prefixes, download_queue, counters))

        download_tasks = [
            asyncio.create_task(download_worker(bucket_name, download_queue, process_queue, session, counters))
            for _ in range(MAX_DOWNLOAD_CONCURRENCY)
        ]

        with ProcessPoolExecutor(max_workers=MAX_PROCESS_WORKERS) as pool:
            process_consumers = [
                asyncio.create_task(
                    process_consumer(process_queue, out_queue, pool, fields, lat_bounds, lon_bounds, quality_max, counters)
                )
                for _ in range(min(MAX_PROCESS_WORKERS, 8))
            ]

            await producer_task

            for _ in range(MAX_DOWNLOAD_CONCURRENCY):
                await download_queue.put(None)

            await download_queue.join()
            await asyncio.gather(*download_tasks)

            for _ in range(len(process_consumers)):
                await process_queue.put(None)

            await process_queue.join()
            await asyncio.gather(*process_consumers)

    await out_queue.put(None)
    await out_queue.join()
    await writer_task

    print("\nFINAL:", counters)

    progress_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await progress_task

    print()
    return output_csv_path


def main():
    bucket_name = "gcp-public-data-goes-16"
    start_date = "2024-01-01"
    end_date = "2024-02-02"

    fields = [
        "flash_id",
        "flash_time_offset_of_first_event",
        "product_time",
        "flash_lat",
        "flash_lon",
        "flash_quality_flag",
        "flash_energy",
        # product_time_offset is added automatically
    ]

    lat_bounds = (0.0, 55.0)
    lon_bounds = (-135.0, -45.0)

    prefixes = generate_prefixes(start_date, end_date)

    output_path = asyncio.run(
        main_async(bucket_name, prefixes, fields, lat_bounds, lon_bounds, quality_max=1, output_csv_path=OUTPUT_CSV_PATH)
    )

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"Data saved to {output_path}")
    else:
        print("No data written (or all rows filtered out).")


if __name__ == "__main__":
    main()
