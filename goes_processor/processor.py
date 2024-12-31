import pandas as pd
import netCDF4 as nc

async def process_worker(process_queue, fields, lat_bounds, lon_bounds, accumulated_data, process_counter):
    while True:
        file_data = await process_queue.get()
        if file_data is None: break
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