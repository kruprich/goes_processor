import numpy as np
import pandas as pd
import xarray as xr

from satpy import Scene
from pyresample.bucket import BucketResampler


def _get_abi_area_from_file(abi_l2_path: str, dataset_name: str = "HT"):
    """
    Load one ABI L2 file with Satpy and grab the fixed-grid AreaDefinition
    from a dataset (e.g. ACHA's 'HT').
    """
    scn = Scene(reader="abi_l2_nc", filenames=[abi_l2_path])
    scn.load([dataset_name])  # loads just enough to attach the 'area' metadata
    da = scn[dataset_name]
    area = da.attrs["area"]   # pyresample AreaDefinition (fixed grid)
    return area, da


def grid_csv_to_abi(
    csv_path: str,
    abi_l2_path: str,
    dataset_name: str = "HT",     # for ACHA this is typically HT
    t0: str | None = None,        # optional ISO timestamp filter start
    t1: str | None = None,        # optional ISO timestamp filter end
    qmax: int = 0,                # you said you want DQF==0 analog; for GLM quality use <=0
    roi_m: float = 8000.0,        # radius-of-influence in meters (tolerance for geo error)
):
    # 1) Get target ABI fixed-grid from Satpy
    target_area, abi_da = _get_abi_area_from_file(abi_l2_path, dataset_name=dataset_name)
    ny, nx = target_area.shape  # (rows, cols)

    # 2) Read CSV
    df = pd.read_csv(csv_path, parse_dates=["product_time_offset"])

    # 3) Filter by time window if provided
    if t0 is not None:
        df = df[df["product_time_offset"] >= pd.Timestamp(t0)]
    if t1 is not None:
        df = df[df["product_time_offset"] < pd.Timestamp(t1)]

    # 4) Filter by flash quality
    # NOTE: this is GLM flash_quality_flag (not ABI DQF). Keep best only.
    df = df[df["flash_quality_flag"] <= qmax]

    if df.empty:
        # Return empty grids
        coords = {"y": np.arange(ny), "x": np.arange(nx)}
        return xr.Dataset(
            {
                "any_flash": (("y", "x"), np.zeros((ny, nx), dtype=np.uint8)),
                "flash_count": (("y", "x"), np.zeros((ny, nx), dtype=np.int32)),
                "energy_sum": (("y", "x"), np.zeros((ny, nx), dtype=np.float32)),
            },
            coords=coords,
            attrs={"abi_file": abi_l2_path, "note": "no points after filtering"},
        )

    # 5) Bucket-resample points onto ABI grid
    lons = df["flash_lon"].to_numpy(dtype=np.float64)
    lats = df["flash_lat"].to_numpy(dtype=np.float64)

    # BucketResampler expects (target_area, source_lons, source_lats)
    br = BucketResampler(target_area, lons, lats)

    ones = np.ones(df.shape[0], dtype=np.float32)
    flash_count = br.get_sum(ones, radius_of_influence=roi_m)  # count per cell

    # Optional: sum of flash_energy per cell
    energy = df["flash_energy"].to_numpy(dtype=np.float32)
    energy_sum = br.get_sum(energy, radius_of_influence=roi_m)

    any_flash = (flash_count > 0).astype(np.uint8)

    # 6) Package as xarray, aligned to ABI grid shape
    coords = {"y": np.arange(ny), "x": np.arange(nx)}
    out = xr.Dataset(
        {
            "any_flash": (("y", "x"), any_flash),
            "flash_count": (("y", "x"), flash_count.astype(np.int32)),
            "energy_sum": (("y", "x"), energy_sum.astype(np.float32)),
        },
        coords=coords,
        attrs={
            "abi_file": abi_l2_path,
            "dataset_name": dataset_name,
            "roi_m": float(roi_m),
            "qmax": int(qmax),
        },
    )

    return out