from __future__ import annotations

import numpy as np
import zarr

# NOTE: this module assumes these globals exist in the importing module:
# cfg, NUM_FEATURE_CHANNELS, FEATURE_CHANNEL_NAMES, NUM_5MIN_BINS_PER_DAY,
# NUM_ABI_PRODUCTS, ABI_PRODUCT_KEYS


def create_zarr_datasets_and_metadata(
    root,
    *,
    grid_y: int,
    grid_x: int,
    bin_start_times_ns: np.ndarray,
    compressor,
    cfg,
    NUM_FEATURE_CHANNELS: int,
    FEATURE_CHANNEL_NAMES: np.ndarray,
    NUM_5MIN_BINS_PER_DAY: int,
    NUM_ABI_PRODUCTS: int,
    ABI_PRODUCT_KEYS: list[str],
) -> None:
    """
    Creates all coordinate/label datasets, data arrays, and per-array attributes
    inside an already-created Zarr group `root`.

    This is the "schema block" extracted from init_daily_zarr_store_atomic.
    """

    # Coordinate/label datasets
    root.create_dataset(
        "features/channel_name",
        shape=(NUM_FEATURE_CHANNELS,),
        dtype="U32",
        chunks=(NUM_FEATURE_CHANNELS,),
        overwrite=True,
    )[:] = FEATURE_CHANNEL_NAMES

    root.create_dataset("time/bin_start_ns", shape=(NUM_5MIN_BINS_PER_DAY,), chunks=(1024,), dtype="i8", overwrite=True)[:] = (
        bin_start_times_ns.astype("datetime64[ns]").astype(np.int64)
    )
    root.create_dataset("time/bin_center_ns", shape=(NUM_5MIN_BINS_PER_DAY,), chunks=(1024,), dtype="i8", overwrite=True)[:] = (
        (bin_start_times_ns + np.timedelta64(150, "s")).astype("datetime64[ns]").astype(np.int64)
    )

    root.create_dataset("abi/product_key", shape=(NUM_ABI_PRODUCTS,), chunks=(NUM_ABI_PRODUCTS,), dtype="U16", overwrite=True)[:] = np.array(
        ABI_PRODUCT_KEYS, dtype="U16"
    )

    root.create_dataset("grid/x_scan_angle_rad", shape=(grid_x,), chunks=(min(grid_x, 4096),), dtype="f8", overwrite=True)
    root.create_dataset("grid/y_scan_angle_rad", shape=(grid_y,), chunks=(min(grid_y, 4096),), dtype="f8", overwrite=True)

    # Legacy monolithic tensor
    root.create_dataset(
        "features/X",
        shape=(NUM_5MIN_BINS_PER_DAY, NUM_FEATURE_CHANNELS, grid_y, grid_x),
        chunks=(cfg.zarr_time_chunk, NUM_FEATURE_CHANNELS, cfg.zarr_chunk_y, cfg.zarr_chunk_x),
        dtype="f4",
        overwrite=True,
        fill_value=np.nan,
        compressor=compressor,
    )

    # ML-friendly separated tensors
    root.create_dataset(
        "abi/product_value",
        shape=(NUM_5MIN_BINS_PER_DAY, NUM_ABI_PRODUCTS, grid_y, grid_x),
        chunks=(cfg.zarr_time_chunk, NUM_ABI_PRODUCTS, cfg.zarr_chunk_y, cfg.zarr_chunk_x),
        dtype="f4",
        overwrite=True,
        fill_value=np.nan,
        compressor=compressor,
    )
    root.create_dataset(
        "abi/valid_pixel_fraction",
        shape=(NUM_5MIN_BINS_PER_DAY, NUM_ABI_PRODUCTS, grid_y, grid_x),
        chunks=(cfg.zarr_time_chunk, NUM_ABI_PRODUCTS, cfg.zarr_chunk_y, cfg.zarr_chunk_x),
        dtype="f4",
        overwrite=True,
        fill_value=0.0,
        compressor=compressor,
    )
    root.create_dataset(
        "abi/bin_has_decoded_file",
        shape=(NUM_5MIN_BINS_PER_DAY, NUM_ABI_PRODUCTS),
        chunks=(NUM_5MIN_BINS_PER_DAY, NUM_ABI_PRODUCTS),
        dtype="u1",
        overwrite=True,
        fill_value=0,
        compressor=compressor,
    )

    root.create_dataset(
        "glm/flash_count",
        shape=(NUM_5MIN_BINS_PER_DAY, grid_y, grid_x),
        chunks=(cfg.zarr_time_chunk, cfg.zarr_chunk_y, cfg.zarr_chunk_x),
        dtype="f4",
        overwrite=True,
        fill_value=0.0,
        compressor=compressor,
    )
    root.create_dataset(
        "glm/bin_has_decoded_file",
        shape=(NUM_5MIN_BINS_PER_DAY,),
        chunks=(NUM_5MIN_BINS_PER_DAY,),
        dtype="u1",
        overwrite=True,
        fill_value=0,
        compressor=compressor,
    )
    root.create_dataset(
        "glm/files_listed_in_bin",
        shape=(NUM_5MIN_BINS_PER_DAY,),
        chunks=(NUM_5MIN_BINS_PER_DAY,),
        dtype="i2",
        overwrite=True,
        fill_value=0,
        compressor=compressor,
    )
    root.create_dataset(
        "glm/files_decoded_ok_in_bin",
        shape=(NUM_5MIN_BINS_PER_DAY,),
        chunks=(NUM_5MIN_BINS_PER_DAY,),
        dtype="i2",
        overwrite=True,
        fill_value=0,
        compressor=compressor,
    )

    # Per-array descriptions (attrs)
    root["abi/product_value"].attrs.update({
        "description": "ABI product value on (possibly coarsened) ABI grid. Each cell is the mean of valid (DQF==0) pixels within that block. NaN where no valid pixels or file missing/failed.",
        "dims": ["time_bin", "abi_product", "y", "x"],
        "dtype": "float32",
    })
    root["abi/valid_pixel_fraction"].attrs.update({
        "description": "Fraction of contributing raw pixels with DQF==0 within each cell/block. 0 where file missing/failed or where no pixels are valid.",
        "dims": ["time_bin", "abi_product", "y", "x"],
        "dtype": "float32",
        "range": [0.0, 1.0],
    })
    root["abi/bin_has_decoded_file"].attrs.update({
        "description": "1 if the selected ABI file for (time_bin,abi_product) downloaded and decoded successfully, else 0.",
        "dims": ["time_bin", "abi_product"],
        "dtype": "uint8",
    })
    root["glm/flash_count"].attrs.update({
        "description": "GLM flash counts mapped onto ABI grid for each 5-min time bin. Sum across all GLM files strictly within the bin window. Only flashes with flash_quality_flag==0 are counted.",
        "dims": ["time_bin", "y", "x"],
        "dtype": "float32",
    })
    root["glm/bin_has_decoded_file"].attrs.update({
        "description": "1 if any GLM file in the time bin downloaded and decoded successfully, else 0.",
        "dims": ["time_bin"],
        "dtype": "uint8",
    })
    root["glm/files_listed_in_bin"].attrs.update({
        "description": "Number of GLM files whose start/end times place them strictly in the bin window (capped to int16).",
        "dims": ["time_bin"],
        "dtype": "int16",
    })
    root["glm/files_decoded_ok_in_bin"].attrs.update({
        "description": "Number of GLM files in the bin that successfully decoded (capped to int16).",
        "dims": ["time_bin"],
        "dtype": "int16",
    })
    root["features/X"].attrs.update({
        "description": "Legacy stacked feature tensor for CNNs. Channel meanings given by features/channel_name.",
        "dims": ["time_bin", "channel", "y", "x"],
        "dtype": "float32",
    })
    root["grid/x_scan_angle_rad"].attrs.update({
        "description": "ABI fixed-grid scan-angle x coordinate (radians). Coarsened if coarsen_factor>1.",
        "dims": ["x"],
        "dtype": "float64",
        "units": "radian",
    })
    root["grid/y_scan_angle_rad"].attrs.update({
        "description": "ABI fixed-grid scan-angle y coordinate (radians). Coarsened if coarsen_factor>1.",
        "dims": ["y"],
        "dtype": "float64",
        "units": "radian",
    })
    root["time/bin_start_ns"].attrs.update({
        "description": "UTC bin start time (nanoseconds since epoch).",
        "dims": ["time_bin"],
        "dtype": "int64",
    })
    root["time/bin_center_ns"].attrs.update({
        "description": "UTC bin center time (nanoseconds since epoch).",
        "dims": ["time_bin"],
        "dtype": "int64",
    })
    root["abi/product_key"].attrs.update({
        "description": "Short keys for ABI products along the abi_product dimension.",
        "dims": ["abi_product"],
        "dtype": "str",
    })
    root["features/channel_name"].attrs.update({
        "description": "Names for channels in features/X along the channel dimension.",
        "dims": ["channel"],
        "dtype": "str",
    })
