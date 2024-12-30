from datetime import datetime, timedelta
import os
import sys
import asyncio
import aiohttp
import numpy as np
from io import BytesIO
from multiprocessing import Value, Lock
from google.cloud import storage
import pandas as pd
import nest_asyncio
import netCDF4 as nc
from pyproj import Proj
import xarray as xr

# Apply this only in environments like Colab where an event loop may already be running
nest_asyncio.apply()

# Constants
RETRIES = 3
MAX_CONCURRENT_DOWNLOADS = 16
MAX_CONCURRENT_CONNECTIONS = MAX_CONCURRENT_DOWNLOADS * 2
TIMEOUT_SECONDS = 30
PREFETCH_THREADS = 1
PROCESS_THREADS = 4

# Initialize Google Cloud Storage client (Anonymous Access)
client = storage.Client.create_anonymous_client()

async def download_blob_to_memory_async(bucket_name, blob_name, session, retries=RETRIES):
    url = f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
    for attempt in range(retries):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=TIMEOUT_SECONDS)) as response:
                if response.status == 200:
                    file_data = await response.read()
                    return BytesIO(file_data), len(file_data)
                print(f"Retry {attempt+1}/{retries} failed for {blob_name}: Status {response.status}")
        except (asyncio.TimeoutError, Exception) as e:
            print(f"Error on attempt {attempt+1} for {blob_name}: {e}")
    return None, 0

async def producer_download(bucket_name, download_queue, process_queue, session, download_counter, download_error_counter, downloaded_size, downloaded_size_lock):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

    async def download_task(blob_name):
        async with semaphore:
            file_data, file_size = await download_blob_to_memory_async(bucket_name, blob_name, session)
            if file_data:
                await process_queue.put(file_data)
                with download_counter.get_lock():
                    download_counter.value += 1
                with downloaded_size_lock:
                    downloaded_size.value += file_size
            else:
                with download_error_counter.get_lock():
                    download_error_counter.value += 1

    tasks = []
    while True:
        blob_name = await download_queue.get()
        if blob_name is None:
            break
        tasks.append(asyncio.create_task(download_task(blob_name)))
        download_queue.task_done()

    await asyncio.gather(*tasks)
    await process_queue.put("DONE")

async def consumer_process(fields, lat_bounds, lon_bounds, process_queue, accumulated_data, process_counter, download_error_counter, out_of_range_counter, filtered_size, process_semaphore):
    async def process_task(file_data):
        nonlocal accumulated_data
        async with process_semaphore:
            accumulated_data, processed_bytes = process_single_file(file_data, fields, lat_bounds, lon_bounds, accumulated_data, process_counter, download_error_counter, out_of_range_counter)
            with filtered_size.get_lock():
                filtered_size.value += processed_bytes

    tasks = []
    while True:
        file_data = await process_queue.get()
        if file_data == "DONE": break
        tasks.append(asyncio.create_task(process_task(file_data)))
        process_queue.task_done()

    await asyncio.gather(*tasks)
    return accumulated_data

def generate_julian_days(start_date, end_date):
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    return [(start_dt + timedelta(days=i)).strftime('%j') for i in range((datetime.strptime(end_date, '%Y-%m-%d') - start_dt).days + 1)]

def process_single_file(file_obj, fields, lat_bounds, lon_bounds, accumulated_data, process_counter, download_error_counter, out_of_range_counter):
    processed_bytes = 0
    try:
        # Step 1: Open the NetCDF file using netCDF4 from the BytesIO object
        with nc.Dataset('in_memory.nc', mode='r', memory=file_obj.read()) as ds:
            # Step 2: Extract the projection information
            goes_proj = ds['goes_imager_projection']
            x = ds['x'][:]
            y = ds['y'][:]

            # Projection information from GOES-R ABI fixed grid
            sat_height = goes_proj.perspective_point_height
            long_proj_origin = goes_proj.longitude_of_projection_origin
            semi_major_axis = goes_proj.semi_major_axis
            semi_minor_axis = goes_proj.semi_minor_axis

            # Step 3: Create a projection object using pyproj
            proj = Proj(proj='geos', h=sat_height, lon_0=long_proj_origin, 
                        a=semi_major_axis, b=semi_minor_axis)

            # Step 4: Convert projection coordinates to latitude and longitude
            x_mesh, y_mesh = np.meshgrid(x, y)
            lon, lat = proj(x_mesh, y_mesh, inverse=True)

            # Step 5: Extract relevant fields (TPW, DQF) from the dataset
            tpw_raw = ds['TPW'][:]
            dqf = ds['DQF_Overall'][:]  # Assuming DQF field name here

            # Step 6: Apply scale factor and offset to TPW
            tpw_scale_factor = ds['TPW'].scale_factor
            tpw_add_offset = ds['TPW'].add_offset
            tpw = (tpw_raw * tpw_scale_factor) + tpw_add_offset

            # Create a pandas DataFrame from the extracted data
            df = pd.DataFrame({
                'lat': lat.flatten(),
                'lon': lon.flatten(),
                'TPW': tpw.flatten(),
                'DQF': dqf.flatten()
            })

            if df.empty:
                with download_error_counter.get_lock():
                    download_error_counter.value += 1

            # Step 6: Apply the filtering based on lat/lon and quality flag
            filtered_df = df.loc[
                (df['lat'] >= lat_bounds[0]) & (df['lat'] <= lat_bounds[1]) &
                (df['lon'] >= lon_bounds[0]) & (df['lon'] <= lon_bounds[1]) &
                (df['DQF'] == 0)  # Only include good-quality data
            ].copy()

            if filtered_df.empty:
                with out_of_range_counter.get_lock():
                    out_of_range_counter.value += 1
                return accumulated_data, processed_bytes

            # Step 7: Calculate the midpoint of `time_bounds` and assign it to all rows
            time_bounds = ds['time_bounds'][:]
            time_midpoint = np.mean(time_bounds)

            # Convert the midpoint time to datetime
            reference_time = datetime(2000, 1, 1, 12, 0, 0)
            midpoint_time = pd.to_datetime(time_midpoint, unit='s', origin=reference_time)

            # Step 8: Add the midpoint time to the filtered DataFrame
            filtered_df['time'] = midpoint_time

            # Step 9: Select only the required columns: lat, lon, TPW, DQF, and time
            filtered_df = filtered_df[['lat', 'lon', 'TPW', 'time', 'DQF']]

            # Step 10: Convert DataFrame back to numpy and concatenate with accumulated data
            filtered_data = filtered_df.to_numpy()
            accumulated_data = filtered_data if accumulated_data is None else np.vstack((accumulated_data, filtered_data))

            # Track the number of processed bytes
            processed_bytes = filtered_data.nbytes

            # Step 11: Increment the process counter
            with process_counter.get_lock():
                process_counter.value += 1

            return accumulated_data, processed_bytes

    except Exception as e:
        print(f"Error processing file: {e}")
        return accumulated_data, processed_bytes

async def prefetch_files_to_queue(bucket_name, prefixes, download_queue, total_files_counter, prefetch_semaphore):
    async def prefetch_task(prefix):
        bucket = client.bucket(bucket_name)
        async with prefetch_semaphore:
            blobs = list(bucket.list_blobs(prefix=prefix))
            for blob in blobs:
                await download_queue.put(blob.name)
                with total_files_counter.get_lock():
                    total_files_counter.value += 1

    prefetch_tasks = [asyncio.create_task(prefetch_task(prefix)) for prefix in prefixes]

    await asyncio.gather(*prefetch_tasks)
    await download_queue.put(None)

async def print_stats(download_counter, process_counter, download_error_counter, out_of_range_counter, total_files_counter, filtered_size, downloaded_size):
    last_download_count = 0
    last_process_count = 0
    last_downloaded_size = 0
    start_time = datetime.now()

    while True:
        await asyncio.sleep(1)
        current_time = datetime.now()
        elapsed_time = 1

        download_rate = (download_counter.value - last_download_count) / elapsed_time
        process_rate = (process_counter.value - last_process_count) / elapsed_time
        download_speed = (downloaded_size.value - last_downloaded_size) / elapsed_time / 1e6
        files_left = total_files_counter.value - process_counter.value
        time_to_completion = files_left / process_rate if process_rate > 0 else float('inf')

        if time_to_completion == float('inf'):
            formatted_time_to_completion = "∞"
        else:
            hours, remainder = divmod(int(time_to_completion), 3600)
            minutes, seconds = divmod(remainder, 60)
            formatted_time_to_completion = f"{hours:02}:{minutes:02}:{seconds:02}"

        total_elapsed_time = (current_time - start_time).total_seconds()
        elapsed_hours, elapsed_remainder = divmod(int(total_elapsed_time), 3600)
        elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
        formatted_elapsed_time = f"{elapsed_hours:02}:{elapsed_minutes:02}:{elapsed_seconds:02}"

        last_download_count = download_counter.value
        last_process_count = process_counter.value
        last_downloaded_size = downloaded_size.value

        status_message = (
            f"Elapsed Time:       {formatted_elapsed_time}\n"
            f"Downloaded Files:   {download_counter.value}/{total_files_counter.value} | {download_rate:.2f} files/sec | {download_speed:.2f} MB/s | Time Remaining: {formatted_time_to_completion}\n"
            f"Processed Files:    {process_counter.value}/{total_files_counter.value} | {process_rate:.2f} files/sec | Errors: {download_error_counter.value} | Out of Range: {out_of_range_counter.value}\n"
        )

        sys.stdout.write("\033[F" * (status_message.count("\n")) + "\033[K")
        sys.stdout.write(status_message)
        sys.stdout.flush()

async def main_async(bucket_name, prefixes, fields_to_extract, lat_bounds, lon_bounds):
    download_queue = asyncio.Queue()
    process_queue = asyncio.Queue()
    counters = [Value('i', 0) for _ in range(7)]
    download_counter, process_counter, download_error_counter, out_of_range_counter, total_files_counter, filtered_size, downloaded_size = counters
    downloaded_size_lock = Lock()
    accumulated_data = None

    prefetch_semaphore = asyncio.Semaphore(PREFETCH_THREADS)
    process_semaphore = asyncio.Semaphore(PROCESS_THREADS)

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_CONNECTIONS, limit_per_host=MAX_CONCURRENT_DOWNLOADS, keepalive_timeout=60)
    async with aiohttp.ClientSession(connector=connector) as session:
        stats_task = asyncio.create_task(print_stats(download_counter, process_counter, download_error_counter, out_of_range_counter, total_files_counter, filtered_size, downloaded_size))
        prefetch_task = asyncio.create_task(prefetch_files_to_queue(bucket_name, prefixes, download_queue, total_files_counter, prefetch_semaphore))
        producer_task = asyncio.create_task(producer_download(bucket_name, download_queue, process_queue, session, download_counter, download_error_counter, downloaded_size, downloaded_size_lock))
        consumer_task = asyncio.create_task(consumer_process(fields_to_extract, lat_bounds, lon_bounds, process_queue, accumulated_data, process_counter, download_error_counter, out_of_range_counter, filtered_size, process_semaphore))

        await asyncio.gather(prefetch_task, producer_task, consumer_task)
        stats_task.cancel()

    return await consumer_task

def aggregate_over_time_and_grid(accumulated_data, lat_bounds, lon_bounds):
    # Create a pandas DataFrame from accumulated data
    df = pd.DataFrame(accumulated_data, columns=['lat', 'lon', 'TPW', 'time', 'DQF'])

    # Convert 'time' to datetime and handle any missing or NaT values
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.dropna(subset=['time'])  # Drop rows where 'time' is NaT

    # Align 'time' to 00:00:00 of the day and ensure it starts at the beginning of the day
    df['time'] = df['time'].dt.floor('D') + pd.to_timedelta((df['time'].dt.hour * 60 + df['time'].dt.minute) // 30 * 30, unit='m')

    # Resample the data into 30-minute intervals starting from 00:00:00 using 'min' instead of 'T'
    df = df.set_index('time').resample('30min').mean().reset_index()

    # Create a grid of 10km resolution (no meshgrid needed)
    lat_grid = np.arange(lat_bounds[0], lat_bounds[1], 0.1)  # Roughly 10km at the equator
    lon_grid = np.arange(lon_bounds[0], lon_bounds[1], 0.1)

    # Assign each point to the nearest grid point
    df['lat_grid'] = np.digitize(df['lat'], lat_grid) - 1
    df['lon_grid'] = np.digitize(df['lon'], lon_grid) - 1
    df['lat_grid'] = lat_grid[df['lat_grid']]
    df['lon_grid'] = lon_grid[df['lon_grid']]

    # Group by the grid points and time, take the mean of TPW values
    aggregated_df = df.groupby(['lat_grid', 'lon_grid', pd.Grouper(key='time')]).agg({
        'TPW': 'mean'
    }).reset_index()

    return aggregated_df

def main():
    is_colab = 'google.colab' in sys.modules
    if is_colab:
        from google.colab import drive
        client = storage.Client()
        print("Running in Colab. Mounting Google Drive...")
        drive.mount('/content/drive', force_remount=True)
        drive_path = '/content/drive/My Drive/'
    else:
        drive_path = os.getcwd()

    bucket_name = 'gcp-public-data-goes-16'
    start_date = '2024-01-01'
    end_date = '2024-09-01'

    lat_bounds, lon_bounds = [0, 55], [-135, -45]
    fields_to_extract = ['TPW', 'lat', 'lon','DQF', 'time_bounds']
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    try:
        while current_date <= end_dt:
            current_start_date = current_date.strftime('%Y-%m-%d')
            current_end_date = current_date.strftime('%Y-%m-%d')

            print(f"Processing data for {current_start_date}")

            prefixes = [f'ABI-L2-TPWC/{year}/{julian_day}/' for year in range(int(current_start_date[:4]), int(current_end_date[:4]) + 1)
                        for julian_day in generate_julian_days(current_start_date, current_end_date)]

            loop = asyncio.get_event_loop()
            accumulated_data = loop.run_until_complete(main_async(bucket_name, prefixes, fields_to_extract, lat_bounds, lon_bounds))

            if accumulated_data is not None:
                print("PROCESSING")
                columns = fields_to_extract
                df = pd.DataFrame(accumulated_data, columns=columns)
                final_df = aggregate_over_time_and_grid(df, lat_bounds, lon_bounds)
                csv_file_path = os.path.join(drive_path, f'ABI_L2_TWPC_{current_start_date}_extracted_data.csv')
                final_df.to_csv(csv_file_path, index=False)
                print(f"Processing complete. Combined data saved to {csv_file_path}")
            else:
                print("No data processed.")

            current_date += timedelta(days=1)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()