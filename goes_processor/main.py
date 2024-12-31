import asyncio
import aiohttp
import pandas as pd
from config import *
from utils import generate_prefixes
from downloader import download_worker
from processor import process_worker
from multiprocessing import Value

from google.cloud import storage

# Initialize Google Cloud Storage client (Anonymous Access)
client = storage.Client.create_anonymous_client()

async def main_async(bucket_name, prefixes):
    print("main async started")
    download_queue = asyncio.Queue()
    process_queue = asyncio.Queue()
    download_counter, error_counter, process_counter = Value('i', 0), Value('i', 0), Value('i', 0)
    accumulated_data = []

    # total_files = sum(1 for prefix in prefixes for blob in storage.Client().bucket(bucket_name).list_blobs(prefix=prefix))

    for prefix in prefixes:
        bucket = client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            await download_queue.put(blob.name)

    for _ in range(MAX_DOWNLOAD_CONCURRENCY): await download_queue.put(None)

    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(download_worker(bucket_name, download_queue, process_queue, session, download_counter, error_counter)),
            asyncio.create_task(process_worker(process_queue, FIELDS, LAT_BOUNDS, LON_BOUNDS, accumulated_data, process_counter))
        ]
        await asyncio.gather(*tasks)

    return pd.concat(accumulated_data) if accumulated_data else None

def main():
    print('running.........................')
    prefixes = generate_prefixes("2024-01-01", "2024-01-02")
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(main_async("gcp-public-data-goes-16", prefixes))
    if result is not None:
        result.to_csv("data/filtered_data.csv", index=False)
        print("Filtered data saved!")

main()