import aiohttp
from io import BytesIO
from google.cloud import storage

async def download_blob(session, url, retries, timeout):
    for attempt in range(retries):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                if response.status == 200:
                    return BytesIO(await response.read())
                print(f"Retry {attempt+1}/{retries} for {url}: Status {response.status}")
        except Exception as e:
            print(f"Error on attempt {attempt+1} for {url}: {e}")
    return None

async def download_worker(bucket_name, download_queue, process_queue, session, download_counter, error_counter):
    client = storage.Client.create_anonymous_client()
    while True:
        blob_name = await download_queue.get()
        if blob_name is None: break
        url = f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
        file_data = await download_blob(session, url, retries=3, timeout=30)
        if file_data:
            await process_queue.put(file_data)
            with download_counter.get_lock():
                download_counter.value += 1
        else:
            with error_counter.get_lock():
                error_counter.value += 1
        download_queue.task_done()
    await process_queue.put(None)