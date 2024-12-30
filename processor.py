import asyncio
import aiohttp
from google.cloud import storage
from datetime import datetime, timedelta
from io import BytesIO
import pandas as pd
import numpy as np
from multiprocessing import Value, Lock
import sys
import netCDF4 as nc  # Ensure this is imported for processing

# Constants
RETRIES = 3
MAX_DOWNLOAD_CONCURRENCY = 64
MAX_PROCESS_CONCURRENCY = 16
TIMEOUT_SECONDS = 30

# Initialize Google Cloud Storage client (Anonymous Access)
client = storage.Client.create_anonymous_client()

async def download_blob(session, url, retries=RETRIES):
    """Download a blob asynchronously with retries."""
    for attempt in range(retries):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=TIMEOUT_SECONDS)) as response:
                if response.status == 200:
                    return BytesIO(await response.read())
                print(f"Retry {attempt+1}/{retries} for {url}: Status {response.status}")
        except Exception as e:
            print(f"Error on attempt {attempt+1} for {url}: {e}")
    return None

async def download_worker(bucket_name, download_queue, process_queue, session, download_counter, error_counter):
    """Download files from GCS and put them in the process queue."""
    while True:
        blob_name = await download_queue.get()
        if blob_name is None:  # Sentinel value to stop
            break
        url = f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
        file_data = await download_blob(session, url)
        if file_data:
            await process_queue.put(file_data)
            with download_counter.get_lock():
                download_counter.value += 1
        else:
            with error_counter.get_lock():
                error_counter.value += 1
        download_queue.task_done()
    # Signal processing workers to stop
    await process_queue.put(None)

async def process_worker(process_queue, fields, lat_bounds, lon_bounds, accumulated_data, process_counter):
    """Process NetCDF files and accumulate filtered data."""
    while True:
        file_data = await process_queue.get()
        if file_data is None:  # Sentinel value to stop
            break
        try:
            with nc.Dataset('in_memory.nc', mode='r', memory=file_data.read()) as ds:
                data = {field: ds.variables[field][:] for field in fields if field in ds.variables}
                df = pd.DataFrame(data)
                if not df.empty:
                    filtered_df = df[
                        (df['flash_lat'] >= lat_bounds[0]) & (df['flash_lat'] <= lat_bounds[1]) &
                        (df['flash_lon'] >= lon_bounds[0]) & (df['flash_lon'] <= lon_bounds[1]) &
                        (df['flash_quality_flag'] <= 1)
                    ]
                    if not filtered_df.empty:
                        accumulated_data.append(filtered_df)
                        with process_counter.get_lock():
                            process_counter.value += 1
        except Exception as e:
            print(f"Processing error: {e}")
        process_queue.task_done()

def generate_prefixes(start_date, end_date):
    """Generate GCS prefixes for the given date range."""
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    prefixes = []
    while start_dt <= end_dt:
        year = start_dt.year
        julian_day = start_dt.strftime('%j')
        prefixes.append(f"GLM-L2-LCFA/{year}/{julian_day}/")
        start_dt += timedelta(days=1)
    return prefixes

async def print_progress(download_counter, process_counter, error_counter, total_files):
    """Print progress information periodically."""
    while True:
        downloaded = download_counter.value
        processed = process_counter.value
        errors = error_counter.value
        sys.stdout.write(f"\rDownloaded: {downloaded}/{total_files}, Processed: {processed}, Errors: {errors}")
        sys.stdout.flush()
        await asyncio.sleep(.5)

async def main_async(bucket_name, prefixes, fields, lat_bounds, lon_bounds):
    """Main asynchronous workflow."""
    download_queue = asyncio.Queue()
    process_queue = asyncio.Queue()
    download_counter = Value('i', 0)
    error_counter = Value('i', 0)
    process_counter = Value('i', 0)
    accumulated_data = []

    total_files = 0  # Count total files for progress tracking
    for prefix in prefixes:
        bucket = client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            await download_queue.put(blob.name)
            total_files += 1

    # Add sentinel values to signal the end
    for _ in range(MAX_DOWNLOAD_CONCURRENCY):
        await download_queue.put(None)

    # Run tasks
    connector = aiohttp.TCPConnector(limit=MAX_DOWNLOAD_CONCURRENCY)
    async with aiohttp.ClientSession(connector=connector) as session:
        progress_task = asyncio.create_task(print_progress(download_counter, process_counter, error_counter, total_files))
        download_tasks = [
            asyncio.create_task(download_worker(bucket_name, download_queue, process_queue, session, download_counter, error_counter))
            for _ in range(MAX_DOWNLOAD_CONCURRENCY)
        ]
        process_tasks = [
            asyncio.create_task(process_worker(process_queue, fields, lat_bounds, lon_bounds, accumulated_data, process_counter))
            for _ in range(MAX_PROCESS_CONCURRENCY)
        ]
        await asyncio.gather(*download_tasks)
        await asyncio.gather(*process_tasks)
        progress_task.cancel()

    return pd.concat(accumulated_data) if accumulated_data else None

def main():
    bucket_name = "gcp-public-data-goes-16"
    start_date = "2024-01-01"
    end_date = "2024-01-02"
    fields = ['flash_id', 'flash_time_offset_of_first_event', 'product_time', 'flash_lat', 'flash_lon', 'flash_quality_flag', 'flash_energy']
    lat_bounds = [0, 55]
    lon_bounds = [-135, -45]

    prefixes = generate_prefixes(start_date, end_date)

    loop = asyncio.get_event_loop()
    result_df = loop.run_until_complete(main_async(bucket_name, prefixes, fields, lat_bounds, lon_bounds))

    if result_df is not None:
        result_df.to_csv("filtered_data.csv", index=False)
        print("\nData saved to filtered_data.csv")
    else:
        print("\nNo data processed.")

if __name__ == "__main__":
    main()