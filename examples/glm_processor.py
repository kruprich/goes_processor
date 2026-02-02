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
import async_counter
import logging

# Constants
RETRIES = 3
FETCH_CONCURRENCY = 1
MAX_DOWNLOAD_ASYNC_TASKS = 64
MAX_PROCESS_CONCURRENCY = 16
TIMEOUT_SECONDS = 30
LOG_FILE = True


if LOG_FILE:
    # Configure the logger to write to output
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # Log messages will be written to 'app.log'
        logging.StreamHandler()         # Log messages will also be printed to the console
    ]
    )
else:
    #Configure to print errors only
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


logger = logging.getLogger("goes_downloader")

# Initialize Google Cloud Storage client (Anonymous Access)
client = storage.Client.create_anonymous_client()

async def download_blob(session, url, retries=RETRIES):
    """Download a blob asynchronously with retries."""
    for attempt in range(retries):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=TIMEOUT_SECONDS)) as response:
                if response.status == 200:
                    return BytesIO(await response.read())
                logger.warning(f"Retry {attempt+1}/{retries} for {url}: Status {response.status}")
        except Exception as e:
            logger.error(f"MAX RETRIES HIT! Final Error on attempt {attempt+1} for {url}: {e}")
    return None

async def list_available_files(bucket_name, prefixes, download_queue, initial_file_counter, client):
    """List available files for each prefix (folder location) and add them to the download queue."""
    for prefix in prefixes:
        try:
            bucket = client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=prefix)
            blob_count = 0
            for blob in blobs:
                await download_queue.put(blob.name)
                blob_count += 1
            await initial_file_counter.increment(by=blob_count)
        except Exception as e:
            logger.error(f"Unable to list_available_files() for {prefix}: {e}")
            # error_counter.value += 1


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
            logger.error(f"Processing error - File Name: {ds.name} {e}")
        process_queue.task_done()

async def generate_folder_queue(start_date, end_date, product):
    """Generate GCS prefixes for the given date range."""
    folder_queue = asyncio.Queue()
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    while start_dt <= end_dt:
        year = start_dt.year
        julian_day = start_dt.strftime('%j')
        folder_queue.put(f"{product}/{year}/{julian_day}/")
        start_dt += timedelta(days=1)
    return folder_queue

async def print_progress(download_counter, process_counter, error_counter, total_files):
    """Print progress information periodically."""
    while True:
        downloaded = download_counter.value
        processed = process_counter.value
        errors = error_counter.value
        sys.stdout.write(f"\rDownloaded: {downloaded}/{total_files}, Processed: {processed}, Errors: {errors}")
        sys.stdout.flush()
        await asyncio.sleep(.5)

async def main_async(bucket_name, product, fields, lat_bounds, lon_bounds, start_date, end_date, prefixes):
    """Main asynchronous workflow."""
    prefixes = generate_prefixes(start_date, end_date)
    folder_queue = generate_folder_queue(start_date, end_date, product)
    download_queue = asyncio.Queue()
    process_queue = asyncio.Queue()
    initial_file_counter = initial_file_counter()
    # download_counter = Value('i', 0)
    error_counter = async_counter.AsyncCounter()
    # process_counter = Value('i', 0)
    accumulated_data = []

    # Add sentinel values to signal the end
    for _ in range(MAX_DOWNLOAD_ASYNC_TASKS):
        await download_queue.put(None)

    # Run tasks
    connector = aiohttp.TCPConnector(limit=MAX_DOWNLOAD_ASYNC_TASKS)
    async with aiohttp.ClientSession(connector=connector) as session:

        progress_task = asyncio.create_task(print_progress(download_queue, process_queue, error_counter, total_files))

        file_list_task = asyncio.create_task(list_available_files(bucket_name, prefixes, download_queue, initial_file_counter, client))

        download_tasks = [
            asyncio.create_task(download_worker(bucket_name, download_queue, process_queue, session, download_counter, error_counter))
            for _ in range(MAX_DOWNLOAD_ASYNC_TASKS)
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
    product = "GLM-L2-LCFA"

    loop = asyncio.get_event_loop()
    result_df = loop.run_until_complete(main_async(product, bucket_name, fields, lat_bounds, lon_bounds, start_date, end_date))

    if result_df is not None:
        result_df.to_csv("filtered_data.csv", index=False)
        print("\nData saved to filtered_data.csv")
    else:
        print("\nNo data processed.")

if __name__ == "__main__":
    main()