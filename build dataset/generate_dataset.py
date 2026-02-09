"""
GOES ABI (CONUS L2) + GLM (LCFA) -> CONFIGURABLE bin aggregation (N x 5-min bins)
================================================================================

ABI aggregation (raw-stack mean, then coarsen ONCE):

For each bin and ABI product:
  - On the native (raw) grid, stack all qualifying ABI files in that bin.
  - For each raw pixel:
      mean across files, ignoring invalid samples (DQF != 0)
      if no valid samples => NaN
  - Then coarsen ONCE by averaging the already-aggregated raw grid (ignore NaNs).

STRICT bin assignment:
  file goes to bin k iff file_start >= bin_start AND file_end < bin_end

Requested metrics (per your definitions)
---------------------------------------

ABI per-bin, per-product, per-output-pixel:
  expected pixels = expected input pixels per bin * coarsen_factor^2
    -> abi/expected_pixel_count[time_bin, abi_product, y, x] = expected_abi_files_per_bin * (f*f)

  valid pixels = pixels present and not bad DQF
    -> abi/valid_pixel_count[time_bin, abi_product, y, x] = sum(valid samples) within each fxf block across decoded files

  valid pixel fraction = valid pixels / expected pixels
    -> abi/valid_pixel_fraction = valid_pixel_count / expected_pixel_count

ABI per-bin, per-product, file metrics (scaled by f*f as requested):
  expected files = expected ABI files per bin * (f*f)
  actual valid files downloaded = decoded_ok_files_in_bin * (f*f)
  valid file fraction = expected / actual_valid   (NaN if actual_valid==0)

GLM per-bin, file metrics (scaled by f*f as requested):
  expected files = expected GLM files per bin * (f*f)
  actual valid files downloaded = decoded_ok_glm_files_in_bin * (f*f)
  valid file fraction = expected / actual_valid   (NaN if actual_valid==0)

Also outputs:
  glm/flash_count[time_bin,y,x]  float32  (sum of rasterized flash counts across GLM files in bin)
  validation/*_files_listed_in_bin, *_files_decoded_ok_in_bin

Deps:
  pip install aiohttp netCDF4 numpy zarr pyproj
"""

from __future__ import annotations

import os
import re
import json
import shutil
import asyncio
import aiohttp
import numpy as np
import zarr
import time

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from concurrent.futures import ProcessPoolExecutor


# =============================================================================
# Config + product definitions
# =============================================================================

@dataclass(frozen=True)
class RemoteProductSpec:
    short_key: str
    gcs_product_prefix: str
    filename_must_contain: str
    data_variable: str | None = None
    dqf_variable: str | None = None


@dataclass(frozen=True)
class PipelineConfig:
    gcs_bucket: str = "gcp-public-data-goes-19"

    start_date_utc: str = "2025-10-16"
    end_date_utc: str = "2026-02-01"

    # Each output "bin" is N*5min wide
    number_of_5min_bins_per_bin: int = 2

    # Coarsen factor f: output grid is raw grid aggregated in fxf blocks
    coarsen_factor: int | None = 4

    max_concurrent_http_requests: int = 32
    http_timeout_seconds: int = 60
    http_retries: int = 3
    tcp_limit_per_host: int = 96

    max_decode_processes: int = 12
    max_days_in_flight: int = 3
    time_bin_write_window: int = 24

    output_root_dir: str = "./goes_abi_glm"
    overwrite_existing: bool = False

    zarr_time_chunk: int = 16
    zarr_chunk_y: int = 256
    zarr_chunk_x: int = 256

    max_concurrent_list_requests: int = 48
    warn_if_write_slower_than_seconds: float = 20.0


cfg = PipelineConfig()

ABI_CMIP_C13 = RemoteProductSpec(
    short_key="cmip",
    gcs_product_prefix="ABI-L2-CMIPC",
    filename_must_contain="ABI-L2-CMIPC-M6C13",
    data_variable="CMI",
    dqf_variable="DQF",
)

ABI_ACHA_HT = RemoteProductSpec(
    short_key="acha",
    gcs_product_prefix="ABI-L2-ACHAC",
    filename_must_contain="ABI-L2-ACHAC-M6",
    data_variable="HT",
    dqf_variable="DQF",
)

ABI_TPW = RemoteProductSpec(
    short_key="tpw",
    gcs_product_prefix="ABI-L2-TPWC",
    filename_must_contain="ABI-L2-TPWC-M6",
    data_variable="TPW",
    dqf_variable="DQF_Overall",
)

GLM_LCFA = RemoteProductSpec(
    short_key="glm",
    gcs_product_prefix="GLM-L2-LCFA",
    filename_must_contain="GLM-L2-LCFA",
)

ABI_PRODUCT_SPECS = [ABI_CMIP_C13, ABI_ACHA_HT, ABI_TPW]
ABI_PRODUCT_KEYS = ["cmip", "acha", "tpw"]
NUM_ABI_PRODUCTS = 3

BASE_5MIN_BINS_PER_DAY = 288
BASE_BIN_SECONDS = 300

EXPECTED_ABI_FILES_PER_5MIN = 1
EXPECTED_GLM_FILES_PER_5MIN = 15

FEATURE_CHANNEL_NAMES = np.array(
    [
        "cmip_c13_cmi",
        "acha_ht",
        "tpw_tpw",
        "glm_flash_count",
        "cmip_validpixfrac",
        "acha_validpixfrac",
        "tpw_validpixfrac",
        "cmip_present",
        "acha_present",
        "tpw_present",
        "glm_present",
    ],
    dtype="U32",
)
NUM_FEATURE_CHANNELS = int(FEATURE_CHANNEL_NAMES.shape[0])


def effective_coarsen_factor() -> int:
    f = cfg.coarsen_factor
    if f is None:
        return 1
    f = int(f)
    if f < 1:
        raise ValueError("coarsen_factor must be >= 1 (or None)")
    return f


def bins_per_day() -> int:
    k = int(cfg.number_of_5min_bins_per_bin)
    if k < 1:
        raise ValueError("number_of_5min_bins_per_bin must be >= 1")
    if BASE_5MIN_BINS_PER_DAY % k != 0:
        raise ValueError(
            f"number_of_5min_bins_per_bin={k} must divide {BASE_5MIN_BINS_PER_DAY} exactly"
        )
    return BASE_5MIN_BINS_PER_DAY // k


def bin_seconds() -> int:
    return BASE_BIN_SECONDS * int(cfg.number_of_5min_bins_per_bin)


def expected_abi_files_per_bin() -> int:
    return int(cfg.number_of_5min_bins_per_bin) * EXPECTED_ABI_FILES_PER_5MIN


def expected_glm_files_per_bin() -> int:
    return int(cfg.number_of_5min_bins_per_bin) * EXPECTED_GLM_FILES_PER_5MIN


# =============================================================================
# Logging + filesystem helpers
# =============================================================================

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def remove_directory_tree(path: str):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def zarr_store_exists(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, ".zgroup"))


def zarr_store_is_complete(path: str) -> bool:
    if not zarr_store_exists(path):
        return False
    try:
        root = zarr.open_group(path, mode="r")
        return bool(root.attrs.get("complete", False))
    except Exception:
        return False


def mark_zarr_store_complete(root) -> None:
    root.attrs["complete"] = True
    root.attrs["completed_utc"] = datetime.now(timezone.utc).isoformat()


def output_zarr_path_for_day(output_root: str, year: int, julian_day: int) -> str:
    d = os.path.join(output_root, "merged", str(year))
    ensure_dir(d)
    return os.path.join(d, f"{year}{julian_day:03d}.zarr")


def infer_satellite_label_from_bucket(bucket: str) -> str:
    m = re.search(r"goes-(\d+)", bucket)
    if not m:
        return "UNKNOWN"
    return f"GOES-{int(m.group(1))}"


# =============================================================================
# Time helpers + filename parsing
# =============================================================================

GOES_FILENAME_START_TIME_PATTERN = re.compile(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})(\d)?")
GOES_FILENAME_END_TIME_PATTERN   = re.compile(r"_e(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})(\d)?")


def _datetime_utc_from_parts(year: int, jday: int, hh: int, mm: int, ss: int, tenth: int | None) -> datetime:
    base = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=jday - 1, hours=hh, minutes=mm, seconds=ss)
    if tenth is not None:
        base += timedelta(milliseconds=100 * int(tenth))
    return base


def parse_filename_scan_start_time_utc(blob_name: str) -> datetime | None:
    m = GOES_FILENAME_START_TIME_PATTERN.search(blob_name)
    if not m:
        return None
    y, j, hh, mm, ss, tenth = m.groups()
    return _datetime_utc_from_parts(int(y), int(j), int(hh), int(mm), int(ss), int(tenth) if tenth else None)


def parse_filename_scan_end_time_utc(blob_name: str) -> datetime | None:
    m = GOES_FILENAME_END_TIME_PATTERN.search(blob_name)
    if not m:
        return None
    y, j, hh, mm, ss, tenth = m.groups()
    return _datetime_utc_from_parts(int(y), int(j), int(hh), int(mm), int(ss), int(tenth) if tenth else None)


def iter_days_utc(start_date: str, end_date: str):
    a = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    b = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    cur = a
    while cur <= b:
        yield cur
        cur += timedelta(days=1)


def utc_day_start(year: int, julian_day: int) -> datetime:
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=julian_day - 1)


def build_bin_start_times_ns(year: int, julian_day: int) -> np.ndarray:
    n_bins = bins_per_day()
    d0 = utc_day_start(year, julian_day)
    base = np.datetime64(d0.replace(tzinfo=None), "ns")
    return base + (np.arange(n_bins, dtype=np.int64) * np.timedelta64(bin_seconds(), "s"))


# =============================================================================
# HTTP listing (async JSON API)
# =============================================================================

async def _gcs_list_object_names_http(
    session: aiohttp.ClientSession,
    bucket: str,
    prefix: str,
    list_request_semaphore: asyncio.Semaphore,
) -> list[str]:
    out: list[str] = []
    page_token: str | None = None

    base_url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o"
    timeout = aiohttp.ClientTimeout(total=cfg.http_timeout_seconds)
    params = {"prefix": prefix, "fields": "items(name),nextPageToken", "maxResults": "1000"}

    while True:
        if page_token:
            params["pageToken"] = page_token
        else:
            params.pop("pageToken", None)

        async with list_request_semaphore:
            try:
                async with session.get(base_url, params=params, timeout=timeout) as r:
                    if r.status != 200:
                        return out
                    js = await r.json()
            except Exception:
                return out

        for it in js.get("items", []):
            n = it.get("name")
            if n:
                out.append(n)

        page_token = js.get("nextPageToken")
        if not page_token:
            break

    return out


async def list_day_objects_for_product(
    session: aiohttp.ClientSession,
    product_spec: RemoteProductSpec,
    year: int,
    julian_day: int,
    list_request_semaphore: asyncio.Semaphore,
) -> list[tuple[datetime, datetime, str]]:
    prefixes = [f"{product_spec.gcs_product_prefix}/{year}/{julian_day:03d}/{hh:02d}/" for hh in range(24)]
    tasks = [
        asyncio.create_task(_gcs_list_object_names_http(session, cfg.gcs_bucket, pfx, list_request_semaphore))
        for pfx in prefixes
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    names: list[str] = []
    for res in results:
        if isinstance(res, Exception):
            continue
        names.extend(res)

    out: list[tuple[datetime, datetime, str]] = []
    for n in names:
        if product_spec.filename_must_contain not in n:
            continue
        ts = parse_filename_scan_start_time_utc(n)
        te = parse_filename_scan_end_time_utc(n)
        if ts is None or te is None:
            continue
        out.append((ts, te, n))

    out.sort(key=lambda x: x[0])
    return out


# =============================================================================
# STRICT bin assignment
# =============================================================================

def _bin_index_for_time(day_start_utc: datetime, file_start: datetime) -> int:
    return int(((file_start - day_start_utc).total_seconds()) // float(bin_seconds()))


def _bin_start_end(day_start_utc: datetime, bin_index: int) -> tuple[datetime, datetime]:
    t0 = day_start_utc + timedelta(seconds=bin_seconds() * bin_index)
    t1 = t0 + timedelta(seconds=bin_seconds())
    return t0, t1


def group_files_into_bins_strict(
    day_start_utc: datetime,
    file_time_triples: list[tuple[datetime, datetime, str]],
) -> list[list[str]]:
    n_bins = bins_per_day()
    blobs_for_bin: list[list[str]] = [[] for _ in range(n_bins)]

    for file_start, file_end, blob in file_time_triples:
        bi = _bin_index_for_time(day_start_utc, file_start)
        if bi < 0 or bi >= n_bins:
            continue
        t0, t1 = _bin_start_end(day_start_utc, bi)
        if file_start >= t0 and file_end < t1:
            blobs_for_bin[bi].append(blob)

    return blobs_for_bin


# =============================================================================
# Downloads
# =============================================================================

async def download_gcs_object_bytes(session: aiohttp.ClientSession, blob_name: str) -> bytes | None:
    url = f"https://storage.googleapis.com/{cfg.gcs_bucket}/{blob_name}"
    timeout = aiohttp.ClientTimeout(total=cfg.http_timeout_seconds)

    for attempt in range(1, cfg.http_retries + 1):
        try:
            async with session.get(url, timeout=timeout) as r:
                if r.status == 200:
                    return await r.read()
        except Exception:
            pass
        await asyncio.sleep(min(0.25 * attempt, 1.0))

    return None


# =============================================================================
# Array helpers
# =============================================================================

def pad_or_crop_to_shape_2d(a: np.ndarray, target_y: int, target_x: int, *, fill) -> np.ndarray:
    if a.shape == (target_y, target_x):
        return a
    out = np.full((target_y, target_x), fill, dtype=a.dtype)
    yy = min(target_y, a.shape[0])
    xx = min(target_x, a.shape[1])
    out[:yy, :xx] = a[:yy, :xx]
    return out


def squeeze_to_2d(a: np.ndarray, name: str) -> np.ndarray:
    a = np.squeeze(np.asarray(a))
    if a.ndim != 2:
        raise ValueError(f"{name} not 2D after squeeze: shape={a.shape}")
    return a


def squeeze_to_1d(v: np.ndarray, name: str) -> np.ndarray:
    v = np.squeeze(np.asarray(v))
    if v.ndim != 1:
        raise ValueError(f"{name} not 1D after squeeze: shape={v.shape}")
    return v


# =============================================================================
# ABI probe + projection
# =============================================================================

def read_goes_imager_projection_attributes(payload: bytes) -> dict[str, float | str]:
    import netCDF4 as nc

    with nc.Dataset("inmem", mode="r", memory=payload) as ds:
        if "goes_imager_projection" not in ds.variables:
            raise KeyError("ABI file missing goes_imager_projection variable")
        p = ds.variables["goes_imager_projection"]

        def _need(attr_name: str):
            if not hasattr(p, attr_name):
                raise KeyError(f"goes_imager_projection missing attribute {attr_name!r}")
            return getattr(p, attr_name)

        return {
            "projection_origin_longitude_deg": float(_need("longitude_of_projection_origin")),
            "perspective_point_height_m": float(_need("perspective_point_height")),
            "semi_major_axis_m": float(_need("semi_major_axis")),
            "semi_minor_axis_m": float(_need("semi_minor_axis")),
            "sweep_angle_axis": str(_need("sweep_angle_axis")),
        }


def probe_abi_reference_grid_and_projection(payload: bytes, data_variable: str, dqf_variable: str):
    import netCDF4 as nc

    with nc.Dataset("inmem", mode="r", memory=payload) as ds:
        data = squeeze_to_2d(ds.variables[data_variable][:], f"probe:{data_variable}").astype(np.float32, copy=False)

        dqf = np.squeeze(np.asarray(ds.variables[dqf_variable][:]))
        if dqf.ndim == 0:
            dqf = np.full(data.shape, int(dqf.item()), dtype=np.int16)
        else:
            dqf = squeeze_to_2d(dqf, f"probe:{dqf_variable}").astype(np.int16, copy=False)

        if dqf.shape != data.shape:
            raise ValueError(f"probe dqf shape {dqf.shape} != data shape {data.shape}")

        x_scan_angle_rad = squeeze_to_1d(ds.variables["x"][:], "probe:x").astype(np.float64, copy=False)
        y_scan_angle_rad = squeeze_to_1d(ds.variables["y"][:], "probe:y").astype(np.float64, copy=False)

    raw_y, raw_x = int(data.shape[0]), int(data.shape[1])
    geos_projection = read_goes_imager_projection_attributes(payload)
    return raw_y, raw_x, x_scan_angle_rad, y_scan_angle_rad, geos_projection


def build_coarsened_scan_angle_vectors(x_raw: np.ndarray, y_raw: np.ndarray, f: int):
    if f == 1:
        return x_raw.astype(np.float64, copy=False), y_raw.astype(np.float64, copy=False)

    def trim(v: np.ndarray):
        n2 = (v.shape[0] // f) * f
        return v[:n2]

    x = trim(x_raw)
    y = trim(y_raw)
    x_c = x.reshape(x.shape[0] // f, f).mean(axis=1)
    y_c = y.reshape(y.shape[0] // f, f).mean(axis=1)
    return x_c.astype(np.float64), y_c.astype(np.float64)


# =============================================================================
# ABI decode (raw) + per-bin reduce + coarsen
# =============================================================================

def _read_abi_raw_data_and_valid_mask(
    payload: bytes,
    data_variable: str,
    dqf_variable: str,
    reference_raw_y: int,
    reference_raw_x: int
):
    """
    Returns (data_raw, valid_mask_raw) both shaped (reference_raw_y, reference_raw_x).
    valid_mask_raw True where DQF==0.
    """
    import netCDF4 as nc

    with nc.Dataset("inmem", mode="r", memory=payload) as ds:
        data = squeeze_to_2d(ds.variables[data_variable][:], data_variable).astype(np.float32, copy=False)

        dqf = np.squeeze(np.asarray(ds.variables[dqf_variable][:]))
        if dqf.ndim == 0:
            dqf = np.full(data.shape, int(dqf.item()), dtype=np.int16)
        else:
            dqf = squeeze_to_2d(dqf, dqf_variable).astype(np.int16, copy=False)

        if dqf.shape != data.shape:
            if dqf.size == 1:
                dqf = np.full(data.shape, int(dqf.reshape(-1)[0]), dtype=np.int16)
            else:
                raise ValueError(f"{dqf_variable} shape {dqf.shape} != {data_variable} shape {data.shape}")

    data = pad_or_crop_to_shape_2d(data, reference_raw_y, reference_raw_x, fill=np.nan)
    dqf  = pad_or_crop_to_shape_2d(dqf,  reference_raw_y, reference_raw_x, fill=np.int16(1)).astype(np.int16, copy=False)

    valid = (dqf == 0)
    return data, valid


def _coarsen_mean_ignore_nan(raw_mean: np.ndarray, f: int) -> np.ndarray:
    """
    raw_mean: (raw_y, raw_x) float32 with NaNs where invalid
    returns coarse mean ignoring NaN (shape ~ raw_y//f, raw_x//f)
    """
    if f == 1:
        return raw_mean.astype(np.float32, copy=False)

    y2 = (raw_mean.shape[0] // f) * f
    x2 = (raw_mean.shape[1] // f) * f
    a = raw_mean[:y2, :x2].astype(np.float32, copy=False)

    finite = np.isfinite(a)
    count = finite.reshape(y2 // f, f, x2 // f, f).sum(axis=(1, 3)).astype(np.float32, copy=False)

    a0 = np.where(finite, a, 0.0).astype(np.float32, copy=False)
    s = a0.reshape(y2 // f, f, x2 // f, f).sum(axis=(1, 3)).astype(np.float32, copy=False)

    denom = np.maximum(count, 1.0)
    mean = (s / denom).astype(np.float32, copy=False)
    mean = np.where(count > 0, mean, np.nan).astype(np.float32, copy=False)
    return mean


def _coarsen_sum_int(raw_count: np.ndarray, f: int) -> np.ndarray:
    """
    raw_count: (raw_y, raw_x) int32 counts per raw pixel
    returns coarse sum per block
    """
    if f == 1:
        return raw_count
    y2 = (raw_count.shape[0] // f) * f
    x2 = (raw_count.shape[1] // f) * f
    a = raw_count[:y2, :x2]
    return a.reshape(y2 // f, f, x2 // f, f).sum(axis=(1, 3))


def decode_abi_file_to_raw_contrib(
    payload: bytes,
    data_variable: str,
    dqf_variable: str,
    reference_raw_y: int,
    reference_raw_x: int
):
    """
    Returns:
      raw_sum_valid: float32 (raw_y, raw_x)  (value where valid else 0)
      raw_count_valid: int16 (raw_y, raw_x) (1 where valid else 0)
    """
    data, valid = _read_abi_raw_data_and_valid_mask(payload, data_variable, dqf_variable, reference_raw_y, reference_raw_x)
    v = np.where(valid, data, 0.0).astype(np.float32, copy=False)
    c = valid.astype(np.int16, copy=False)
    return v, c


def _decode_abi_worker(payload: bytes, data_variable: str, dqf_variable: str, reference_raw_y: int, reference_raw_x: int):
    return decode_abi_file_to_raw_contrib(payload, data_variable, dqf_variable, reference_raw_y, reference_raw_x)


# =============================================================================
# GLM -> ABI grid flash_count
# =============================================================================

def rasterize_glm_flash_counts_to_abi_grid(
    payload: bytes,
    x_scan_angle_rad_coarse: np.ndarray,
    y_scan_angle_rad_coarse: np.ndarray,
    geos_projection: dict[str, float | str],
) -> np.ndarray:
    import netCDF4 as nc
    from pyproj import CRS, Transformer

    with nc.Dataset("inmem", mode="r", memory=payload) as ds:
        if ("flash_lat" not in ds.variables) or ("flash_lon" not in ds.variables) or ("flash_quality_flag" not in ds.variables):
            return np.zeros((y_scan_angle_rad_coarse.shape[0], x_scan_angle_rad_coarse.shape[0]), dtype=np.float32)

        lat = np.asarray(ds.variables["flash_lat"][:], dtype=np.float64).reshape(-1)
        lon = np.asarray(ds.variables["flash_lon"][:], dtype=np.float64).reshape(-1)
        q   = np.asarray(ds.variables["flash_quality_flag"][:], dtype=np.int16).reshape(-1)

    if lat.size == 0:
        return np.zeros((y_scan_angle_rad_coarse.shape[0], x_scan_angle_rad_coarse.shape[0]), dtype=np.float32)

    good = (q == 0) & np.isfinite(lat) & np.isfinite(lon)
    if not np.any(good):
        return np.zeros((y_scan_angle_rad_coarse.shape[0], x_scan_angle_rad_coarse.shape[0]), dtype=np.float32)

    lat = lat[good]
    lon = lon[good]

    lon0_deg = float(geos_projection["projection_origin_longitude_deg"])
    h_m      = float(geos_projection["perspective_point_height_m"])
    a_m      = float(geos_projection["semi_major_axis_m"])
    b_m      = float(geos_projection["semi_minor_axis_m"])
    sweep    = str(geos_projection["sweep_angle_axis"])

    geos = CRS.from_proj4(
        f"+proj=geos +lon_0={lon0_deg} +h={h_m} +a={a_m} +b={b_m} +sweep={sweep} +units=m +no_defs"
    )
    wgs84 = CRS.from_epsg(4326)
    tfm = Transformer.from_crs(wgs84, geos, always_xy=True)

    x_m, y_m = tfm.transform(lon, lat)
    x_rad = np.asarray(x_m, dtype=np.float64) / float(h_m)
    y_rad = np.asarray(y_m, dtype=np.float64) / float(h_m)

    xmin, xmax = float(np.min(x_scan_angle_rad_coarse)), float(np.max(x_scan_angle_rad_coarse))
    ymin, ymax = float(np.min(y_scan_angle_rad_coarse)), float(np.max(y_scan_angle_rad_coarse))
    in_bounds = (x_rad >= xmin) & (x_rad <= xmax) & (y_rad >= ymin) & (y_rad <= ymax)
    if not np.any(in_bounds):
        return np.zeros((y_scan_angle_rad_coarse.shape[0], x_scan_angle_rad_coarse.shape[0]), dtype=np.float32)

    x_rad = x_rad[in_bounds]
    y_rad = y_rad[in_bounds]

    x_vec = x_scan_angle_rad_coarse
    x_ascending = x_vec[0] < x_vec[-1]
    if not x_ascending:
        x_vec = x_vec[::-1]

    y_vec = y_scan_angle_rad_coarse
    y_ascending = y_vec[0] < y_vec[-1]
    if not y_ascending:
        y_vec = y_vec[::-1]

    ix = np.searchsorted(x_vec, x_rad, side="left")
    iy = np.searchsorted(y_vec, y_rad, side="left")
    ix = np.clip(ix, 1, x_vec.shape[0] - 1)
    iy = np.clip(iy, 1, y_vec.shape[0] - 1)

    x0 = x_vec[ix - 1]
    x1 = x_vec[ix]
    ix = np.where(np.abs(x_rad - x0) <= np.abs(x_rad - x1), ix - 1, ix)

    y0v = y_vec[iy - 1]
    y1v = y_vec[iy]
    iy = np.where(np.abs(y_rad - y0v) <= np.abs(y_rad - y1v), iy - 1, iy)

    if not x_ascending:
        ix = (x_scan_angle_rad_coarse.shape[0] - 1) - ix
    if not y_ascending:
        iy = (y_scan_angle_rad_coarse.shape[0] - 1) - iy

    out = np.zeros((y_scan_angle_rad_coarse.shape[0], x_scan_angle_rad_coarse.shape[0]), dtype=np.float32)
    np.add.at(out, (iy, ix), 1.0)
    return out


def _decode_glm_worker(payload: bytes, x_scan_angle_rad_coarse: np.ndarray, y_scan_angle_rad_coarse: np.ndarray, geos_projection: dict[str, float | str]):
    return rasterize_glm_flash_counts_to_abi_grid(payload, x_scan_angle_rad_coarse, y_scan_angle_rad_coarse, geos_projection)


# =============================================================================
# Zarr init
# =============================================================================

def init_daily_zarr_store_atomic(
    out_zarr_path: str,
    grid_y: int,
    grid_x: int,
    year: int,
    julian_day: int,
    bin_start_times_ns: np.ndarray,
    geos_projection: dict[str, float | str],
    coarsen_factor: int,
):
    n_bins = bins_per_day()

    if zarr_store_exists(out_zarr_path) and not cfg.overwrite_existing:
        return zarr.open_group(out_zarr_path, mode="r+")

    tmp_path = out_zarr_path + ".tmp"
    if os.path.isdir(tmp_path):
        remove_directory_tree(tmp_path)

    if zarr_store_exists(out_zarr_path) and cfg.overwrite_existing:
        remove_directory_tree(out_zarr_path)

    compressor = zarr.Blosc(cname="zstd", clevel=1, shuffle=zarr.Blosc.BITSHUFFLE)

    store = zarr.DirectoryStore(tmp_path)
    root = zarr.group(store=store, overwrite=True)

    root.attrs.update(
        {
            "schema_version": "v5_requested_pixel_and_file_metrics_cleanup",
            "year": year,
            "julian_day": julian_day,
            "coarsen_factor_effective": int(coarsen_factor),
            "number_of_5min_bins_per_bin": int(cfg.number_of_5min_bins_per_bin),
            "bin_seconds": int(bin_seconds()),
            "num_time_bins_per_day": int(n_bins),
            "abi_bin_assignment_rule": "STRICT [t0,t1): file_start>=t0 and file_end<t1; include ALL qualifying files",
            "abi_aggregation_rule": "RAW grid mean across files ignoring DQF!=0; then coarsen mean ignoring NaN",
            "glm_bin_assignment_rule": "STRICT [t0,t1): file_start>=t0 and file_end<t1; include all qualifying files",
            "expected_pixel_count_definition": "expected_abi_files_per_bin * (f*f)",
            "valid_pixel_count_definition": "sum(valid samples) in each fxf block across decoded files",
            "valid_pixel_fraction_definition": "valid_pixel_count / expected_pixel_count",
            "valid_file_fraction_definition": "expected_files_scaled / actual_valid_files_scaled (NaN if actual==0)",
            "validation_expected_abi_files_per_5min": int(EXPECTED_ABI_FILES_PER_5MIN),
            "validation_expected_glm_files_per_5min": int(EXPECTED_GLM_FILES_PER_5MIN),
            "validation_expected_abi_files_per_bin": int(expected_abi_files_per_bin()),
            "validation_expected_glm_files_per_bin": int(expected_glm_files_per_bin()),
            "legacy_feature_tensor_channels": int(NUM_FEATURE_CHANNELS),
            "zarr_compressor": "blosc:zstd:clevel1:bitshuffle",
            "grid_shape_yx": [int(grid_y), int(grid_x)],
            "zarr_chunking": {
                "time_chunk": int(cfg.zarr_time_chunk),
                "chunk_y": int(cfg.zarr_chunk_y),
                "chunk_x": int(cfg.zarr_chunk_x),
            },
            "abi_products": [p.gcs_product_prefix for p in ABI_PRODUCT_SPECS],
            "abi_data_variables": [p.data_variable for p in ABI_PRODUCT_SPECS],
            "abi_dqf_variables": [p.dqf_variable for p in ABI_PRODUCT_SPECS],
            "glm_product": GLM_LCFA.gcs_product_prefix,
            "source_gcs_bucket": cfg.gcs_bucket,
            "source_satellite_label": infer_satellite_label_from_bucket(cfg.gcs_bucket),
            "goes_geos_projection": geos_projection,
            "complete": False,
            "completed_utc": None,
        }
    )

    # axes
    root.create_dataset("features/channel_name", shape=(NUM_FEATURE_CHANNELS,), dtype="U32", chunks=(NUM_FEATURE_CHANNELS,), overwrite=True)[:] = FEATURE_CHANNEL_NAMES

    root.create_dataset("time/bin_start_ns", shape=(n_bins,), chunks=(min(n_bins, 1024),), dtype="i8", overwrite=True)[:] = (
        bin_start_times_ns.astype("datetime64[ns]").astype(np.int64)
    )
    root.create_dataset("time/bin_center_ns", shape=(n_bins,), chunks=(min(n_bins, 1024),), dtype="i8", overwrite=True)[:] = (
        (bin_start_times_ns + np.timedelta64(bin_seconds() // 2, "s")).astype("datetime64[ns]").astype(np.int64)
    )

    root.create_dataset("abi/product_key", shape=(NUM_ABI_PRODUCTS,), chunks=(NUM_ABI_PRODUCTS,), dtype="U16", overwrite=True)[:] = np.array(
        ABI_PRODUCT_KEYS, dtype="U16"
    )

    root.create_dataset("grid/x_scan_angle_rad", shape=(grid_x,), chunks=(min(grid_x, 4096),), dtype="f8", overwrite=True)
    root.create_dataset("grid/y_scan_angle_rad", shape=(grid_y,), chunks=(min(grid_y, 4096),), dtype="f8", overwrite=True)

    # legacy tensor
    root.create_dataset(
        "features/X",
        shape=(n_bins, NUM_FEATURE_CHANNELS, grid_y, grid_x),
        chunks=(min(cfg.zarr_time_chunk, n_bins), NUM_FEATURE_CHANNELS, cfg.zarr_chunk_y, cfg.zarr_chunk_x),
        dtype="f4",
        overwrite=True,
        fill_value=np.nan,
        compressor=compressor,
    )

    # ABI outputs
    root.create_dataset(
        "abi/product_value",
        shape=(n_bins, NUM_ABI_PRODUCTS, grid_y, grid_x),
        chunks=(min(cfg.zarr_time_chunk, n_bins), NUM_ABI_PRODUCTS, cfg.zarr_chunk_y, cfg.zarr_chunk_x),
        dtype="f4",
        overwrite=True,
        fill_value=np.nan,
        compressor=compressor,
    )
    root.create_dataset(
        "abi/expected_pixel_count",
        shape=(n_bins, NUM_ABI_PRODUCTS, grid_y, grid_x),
        chunks=(min(cfg.zarr_time_chunk, n_bins), 1, cfg.zarr_chunk_y, cfg.zarr_chunk_x),
        dtype="i4",
        overwrite=True,
        fill_value=0,
        compressor=compressor,
    )
    root.create_dataset(
        "abi/valid_pixel_count",
        shape=(n_bins, NUM_ABI_PRODUCTS, grid_y, grid_x),
        chunks=(min(cfg.zarr_time_chunk, n_bins), 1, cfg.zarr_chunk_y, cfg.zarr_chunk_x),
        dtype="i4",
        overwrite=True,
        fill_value=0,
        compressor=compressor,
    )
    root.create_dataset(
        "abi/valid_pixel_fraction",
        shape=(n_bins, NUM_ABI_PRODUCTS, grid_y, grid_x),
        chunks=(min(cfg.zarr_time_chunk, n_bins), NUM_ABI_PRODUCTS, cfg.zarr_chunk_y, cfg.zarr_chunk_x),
        dtype="f4",
        overwrite=True,
        fill_value=0.0,
        compressor=compressor,
    )
    root.create_dataset(
        "abi/bin_has_decoded_file",
        shape=(n_bins, NUM_ABI_PRODUCTS),
        chunks=(n_bins, NUM_ABI_PRODUCTS),
        dtype="u1",
        overwrite=True,
        fill_value=0,
        compressor=compressor,
    )

    # GLM outputs
    root.create_dataset(
        "glm/flash_count",
        shape=(n_bins, grid_y, grid_x),
        chunks=(min(cfg.zarr_time_chunk, n_bins), cfg.zarr_chunk_y, cfg.zarr_chunk_x),
        dtype="f4",
        overwrite=True,
        fill_value=0.0,
        compressor=compressor,
    )
    root.create_dataset(
        "glm/bin_has_decoded_file",
        shape=(n_bins,),
        chunks=(n_bins,),
        dtype="u1",
        overwrite=True,
        fill_value=0,
        compressor=compressor,
    )

    # Validation listed/ok counts
    root.create_dataset(
        "validation/abi_files_listed_in_bin",
        shape=(n_bins, NUM_ABI_PRODUCTS),
        chunks=(n_bins, NUM_ABI_PRODUCTS),
        dtype="i2",
        overwrite=True,
        fill_value=0,
        compressor=compressor,
    )
    root.create_dataset(
        "validation/abi_files_decoded_ok_in_bin",
        shape=(n_bins, NUM_ABI_PRODUCTS),
        chunks=(n_bins, NUM_ABI_PRODUCTS),
        dtype="i2",
        overwrite=True,
        fill_value=0,
        compressor=compressor,
    )
    root.create_dataset(
        "validation/glm_files_listed_in_bin",
        shape=(n_bins,),
        chunks=(n_bins,),
        dtype="i2",
        overwrite=True,
        fill_value=0,
        compressor=compressor,
    )
    root.create_dataset(
        "validation/glm_files_decoded_ok_in_bin",
        shape=(n_bins,),
        chunks=(n_bins,),
        dtype="i2",
        overwrite=True,
        fill_value=0,
        compressor=compressor,
    )

    # Requested scaled file metrics
    root.create_dataset(
        "validation/abi_expected_files_scaled",
        shape=(n_bins, NUM_ABI_PRODUCTS),
        chunks=(n_bins, NUM_ABI_PRODUCTS),
        dtype="f4",
        overwrite=True,
        fill_value=0.0,
        compressor=compressor,
    )
    root.create_dataset(
        "validation/abi_actual_valid_files_scaled",
        shape=(n_bins, NUM_ABI_PRODUCTS),
        chunks=(n_bins, NUM_ABI_PRODUCTS),
        dtype="f4",
        overwrite=True,
        fill_value=0.0,
        compressor=compressor,
    )
    root.create_dataset(
        "validation/abi_valid_file_fraction",
        shape=(n_bins, NUM_ABI_PRODUCTS),
        chunks=(n_bins, NUM_ABI_PRODUCTS),
        dtype="f4",
        overwrite=True,
        fill_value=np.nan,
        compressor=compressor,
    )

    root.create_dataset(
        "validation/glm_expected_files_scaled",
        shape=(n_bins,),
        chunks=(n_bins,),
        dtype="f4",
        overwrite=True,
        fill_value=0.0,
        compressor=compressor,
    )
    root.create_dataset(
        "validation/glm_actual_valid_files_scaled",
        shape=(n_bins,),
        chunks=(n_bins,),
        dtype="f4",
        overwrite=True,
        fill_value=0.0,
        compressor=compressor,
    )
    root.create_dataset(
        "validation/glm_valid_file_fraction",
        shape=(n_bins,),
        chunks=(n_bins,),
        dtype="f4",
        overwrite=True,
        fill_value=np.nan,
        compressor=compressor,
    )

    # Atomic move
    if os.path.isdir(out_zarr_path):
        remove_directory_tree(out_zarr_path)
    shutil.move(tmp_path, out_zarr_path)
    return zarr.open_group(out_zarr_path, mode="r+")


def _write_time_window_to_zarr(
    root,
    window_start_bin: int,
    X_window: np.ndarray,
    abi_value_window: np.ndarray,
    abi_expected_pixel_count_window: np.ndarray,
    abi_valid_pixel_count_window: np.ndarray,
    abi_valid_pixel_fraction_window: np.ndarray,
    abi_bin_present_window: np.ndarray,
    glm_flash_count_window: np.ndarray,
    glm_bin_present_window: np.ndarray,
    glm_files_listed_window: np.ndarray,
    glm_files_ok_window: np.ndarray,
    abi_files_listed_window: np.ndarray,
    abi_files_ok_window: np.ndarray,
    abi_expected_files_scaled_window: np.ndarray,
    abi_actual_valid_files_scaled_window: np.ndarray,
    abi_valid_file_fraction_window: np.ndarray,
    glm_expected_files_scaled_window: np.ndarray,
    glm_actual_valid_files_scaled_window: np.ndarray,
    glm_valid_file_fraction_window: np.ndarray,
):
    window_end_bin = window_start_bin + X_window.shape[0]

    root["features/X"][window_start_bin:window_end_bin, :, :, :] = X_window

    root["abi/product_value"][window_start_bin:window_end_bin, :, :, :] = abi_value_window
    root["abi/expected_pixel_count"][window_start_bin:window_end_bin, :, :, :] = abi_expected_pixel_count_window
    root["abi/valid_pixel_count"][window_start_bin:window_end_bin, :, :, :] = abi_valid_pixel_count_window
    root["abi/valid_pixel_fraction"][window_start_bin:window_end_bin, :, :, :] = abi_valid_pixel_fraction_window
    root["abi/bin_has_decoded_file"][window_start_bin:window_end_bin, :] = abi_bin_present_window

    root["glm/flash_count"][window_start_bin:window_end_bin, :, :] = glm_flash_count_window
    root["glm/bin_has_decoded_file"][window_start_bin:window_end_bin] = glm_bin_present_window

    root["validation/glm_files_listed_in_bin"][window_start_bin:window_end_bin] = glm_files_listed_window
    root["validation/glm_files_decoded_ok_in_bin"][window_start_bin:window_end_bin] = glm_files_ok_window
    root["validation/abi_files_listed_in_bin"][window_start_bin:window_end_bin, :] = abi_files_listed_window
    root["validation/abi_files_decoded_ok_in_bin"][window_start_bin:window_end_bin, :] = abi_files_ok_window

    root["validation/abi_expected_files_scaled"][window_start_bin:window_end_bin, :] = abi_expected_files_scaled_window
    root["validation/abi_actual_valid_files_scaled"][window_start_bin:window_end_bin, :] = abi_actual_valid_files_scaled_window
    root["validation/abi_valid_file_fraction"][window_start_bin:window_end_bin, :] = abi_valid_file_fraction_window

    root["validation/glm_expected_files_scaled"][window_start_bin:window_end_bin] = glm_expected_files_scaled_window
    root["validation/glm_actual_valid_files_scaled"][window_start_bin:window_end_bin] = glm_actual_valid_files_scaled_window
    root["validation/glm_valid_file_fraction"][window_start_bin:window_end_bin] = glm_valid_file_fraction_window


# =============================================================================
# Per-day processing
# =============================================================================

async def process_single_day(
    day_utc: datetime,
    session: aiohttp.ClientSession,
    http_semaphore: asyncio.Semaphore,
    decode_pool: ProcessPoolExecutor,
) -> bool:
    n_bins = bins_per_day()
    year = day_utc.year
    julian_day = int(day_utc.strftime("%j"))
    out_zarr_path = output_zarr_path_for_day(cfg.output_root_dir, year, julian_day)

    if not cfg.overwrite_existing and zarr_store_is_complete(out_zarr_path):
        log(f"SKIP {day_utc.date().isoformat()} ({year}{julian_day:03d}) already merged (complete=true)")
        return True

    if (not cfg.overwrite_existing) and zarr_store_exists(out_zarr_path) and (not zarr_store_is_complete(out_zarr_path)):
        log(f"RETRY {day_utc.date().isoformat()} ({year}{julian_day:03d}) partial zarr; rebuilding")
        remove_directory_tree(out_zarr_path)

    f = effective_coarsen_factor()
    log(f"=== {day_utc.date().isoformat()} ({year}{julian_day:03d}) ABI+GLM bins={n_bins} bin_sec={bin_seconds()} coarsen_f={f} ===")

    day_start = utc_day_start(year, julian_day)
    bin_starts_ns = build_bin_start_times_ns(year, julian_day)

    list_semaphore = asyncio.Semaphore(cfg.max_concurrent_list_requests)
    log("listing objects (async)...")
    t_list0 = time.perf_counter()

    abi_list_tasks = [
        asyncio.create_task(list_day_objects_for_product(session, spec, year, julian_day, list_semaphore))
        for spec in ABI_PRODUCT_SPECS
    ]
    glm_list_task = asyncio.create_task(list_day_objects_for_product(session, GLM_LCFA, year, julian_day, list_semaphore))

    abi_file_triples_per_product = await asyncio.gather(*abi_list_tasks)
    glm_file_triples = await glm_list_task
    log(f"listing done in {time.perf_counter() - t_list0:.2f}s")

    abi_blobs_in_bin_by_product: dict[str, list[list[str]]] = {}
    for spec, triples in zip(ABI_PRODUCT_SPECS, abi_file_triples_per_product):
        triples = triples or []
        abi_blobs_in_bin_by_product[spec.short_key] = group_files_into_bins_strict(day_start, triples) if triples else [[] for _ in range(n_bins)]

    glm_triples = glm_file_triples or []
    glm_blobs_in_bin = group_files_into_bins_strict(day_start, glm_triples) if glm_triples else [[] for _ in range(n_bins)]
    glm_listed_counts = np.array([min(len(x), 32767) for x in glm_blobs_in_bin], dtype=np.int16)

    # Probe reference ABI (to determine grid)
    log("probing ABI reference...")
    probe_payload = None
    probe_spec: RemoteProductSpec | None = None
    probe_blob: str | None = None

    for spec in ABI_PRODUCT_SPECS:
        for bin_list in abi_blobs_in_bin_by_product[spec.short_key]:
            if bin_list:
                probe_blob = bin_list[0]
                probe_spec = spec
                async with http_semaphore:
                    probe_payload = await download_gcs_object_bytes(session, probe_blob)
                if probe_payload:
                    break
        if probe_payload:
            break

    if probe_payload is None or probe_spec is None:
        log("!! no ABI probe downloadable; skipping day")
        return False

    try:
        raw_y, raw_x, x_raw, y_raw, geos_projection = probe_abi_reference_grid_and_projection(
            probe_payload,
            probe_spec.data_variable,  # type: ignore[arg-type]
            probe_spec.dqf_variable,   # type: ignore[arg-type]
        )
        x_scan_angle_rad, y_scan_angle_rad = build_coarsened_scan_angle_vectors(x_raw, y_raw, f)

        grid_y = int(y_scan_angle_rad.shape[0])
        grid_x = int(x_scan_angle_rad.shape[0])

        log(f"probe OK: raw={raw_y}x{raw_x} -> coarse grid={grid_y}x{grid_x} ({os.path.basename(probe_blob or '')})")
    except Exception as e:
        log(f"!! probe failed: {repr(e)}")
        return False

    # Init Zarr
    try:
        root = init_daily_zarr_store_atomic(
            out_zarr_path,
            grid_y=grid_y,
            grid_x=grid_x,
            year=year,
            julian_day=julian_day,
            bin_start_times_ns=bin_starts_ns,
            geos_projection=geos_projection,
            coarsen_factor=f,
        )
    except Exception as e:
        log(f"!! init zarr failed: {repr(e)}")
        return False

    try:
        root["grid/x_scan_angle_rad"][:] = x_scan_angle_rad.astype(np.float64, copy=False)
        root["grid/y_scan_angle_rad"][:] = y_scan_angle_rad.astype(np.float64, copy=False)
    except Exception:
        pass

    loop = asyncio.get_running_loop()

    async def download_and_decode_abi_raw_contrib(bin_index: int, spec: RemoteProductSpec, blob: str):
        async with http_semaphore:
            payload = await download_gcs_object_bytes(session, blob)
        if payload is None:
            return (bin_index, spec.short_key, False, None, None)

        try:
            raw_sum, raw_count = await loop.run_in_executor(
                decode_pool,
                _decode_abi_worker,
                payload,
                spec.data_variable,
                spec.dqf_variable,
                raw_y,
                raw_x,
            )  # type: ignore[arg-type]
            return (bin_index, spec.short_key, True, raw_sum, raw_count)
        except Exception:
            return (bin_index, spec.short_key, False, None, None)

    async def download_and_decode_glm_file_for_bin(bin_index: int, blob: str):
        async with http_semaphore:
            payload = await download_gcs_object_bytes(session, blob)
        if payload is None:
            return (bin_index, False, None)

        try:
            counts = await loop.run_in_executor(
                decode_pool,
                _decode_glm_worker,
                payload,
                x_scan_angle_rad,
                y_scan_angle_rad,
                geos_projection,
            )
            return (bin_index, True, counts)
        except Exception:
            return (bin_index, False, None)

    ones_grid = np.ones((grid_y, grid_x), dtype=np.float32)

    pix_mult = float(f * f)

    expected_pix_per_output_pixel = int(expected_abi_files_per_bin() * (f * f))

    expected_abi_files_scaled = float(expected_abi_files_per_bin()) * pix_mult
    expected_glm_files_scaled = float(expected_glm_files_per_bin()) * pix_mult

    prod_index = {"cmip": 0, "acha": 1, "tpw": 2}

    for window_start in range(0, n_bins, cfg.time_bin_write_window):
        window_end = min(n_bins, window_start + cfg.time_bin_write_window)
        window_len = window_end - window_start

        # ABI window arrays
        abi_value_window = np.full((window_len, NUM_ABI_PRODUCTS, grid_y, grid_x), np.nan, dtype=np.float32)
        abi_expected_pix_window = np.zeros((window_len, NUM_ABI_PRODUCTS, grid_y, grid_x), dtype=np.int32)
        abi_valid_pix_window    = np.zeros((window_len, NUM_ABI_PRODUCTS, grid_y, grid_x), dtype=np.int32)
        abi_vpf_window          = np.zeros((window_len, NUM_ABI_PRODUCTS, grid_y, grid_x), dtype=np.float32)
        abi_present_win         = np.zeros((window_len, NUM_ABI_PRODUCTS), dtype=np.uint8)

        # GLM window arrays
        glm_flash_count_sum = np.zeros((window_len, grid_y, grid_x), dtype=np.float32)
        glm_bin_present = np.zeros((window_len,), dtype=np.uint8)

        # Validation (listed/ok counts)
        glm_files_listed = np.zeros((window_len,), dtype=np.int16)
        glm_files_ok = np.zeros((window_len,), dtype=np.int16)
        abi_files_listed_window = np.zeros((window_len, NUM_ABI_PRODUCTS), dtype=np.int16)
        abi_files_ok_window = np.zeros((window_len, NUM_ABI_PRODUCTS), dtype=np.int16)

        # Requested scaled file metrics
        abi_expected_files_scaled_win = np.full((window_len, NUM_ABI_PRODUCTS), np.float32(expected_abi_files_scaled), dtype=np.float32)
        abi_actual_valid_files_scaled_win = np.zeros((window_len, NUM_ABI_PRODUCTS), dtype=np.float32)
        abi_valid_file_fraction_win = np.full((window_len, NUM_ABI_PRODUCTS), np.nan, dtype=np.float32)

        glm_expected_files_scaled_win = np.full((window_len,), np.float32(expected_glm_files_scaled), dtype=np.float32)
        glm_actual_valid_files_scaled_win = np.zeros((window_len,), dtype=np.float32)
        glm_valid_file_fraction_win = np.full((window_len,), np.nan, dtype=np.float32)

        # Raw accumulators for ABI: per product, per bin
        # (raw_sum float32, raw_count int32, ok_files int)
        raw_acc = {
            "cmip": [(np.zeros((raw_y, raw_x), dtype=np.float32), np.zeros((raw_y, raw_x), dtype=np.int32), 0) for _ in range(window_len)],
            "acha": [(np.zeros((raw_y, raw_x), dtype=np.float32), np.zeros((raw_y, raw_x), dtype=np.int32), 0) for _ in range(window_len)],
            "tpw":  [(np.zeros((raw_y, raw_x), dtype=np.float32), np.zeros((raw_y, raw_x), dtype=np.int32), 0) for _ in range(window_len)],
        }

        # Fill constant expected pixels and listed counts
        for bin_index in range(window_start, window_end):
            wi = bin_index - window_start

            glm_files_listed[wi] = glm_listed_counts[bin_index]

            for key, pidx in prod_index.items():
                abi_files_listed_window[wi, pidx] = np.int16(min(len(abi_blobs_in_bin_by_product[key][bin_index]), 32767))

            abi_expected_pix_window[wi, :, :, :] = np.int32(expected_pix_per_output_pixel)

        # Schedule downloads/decodes
        tasks: list[asyncio.Task] = []
        for bin_index in range(window_start, window_end):
            for spec in ABI_PRODUCT_SPECS:
                for blob in abi_blobs_in_bin_by_product[spec.short_key][bin_index]:
                    tasks.append(asyncio.create_task(download_and_decode_abi_raw_contrib(bin_index, spec, blob)))

            for blob in glm_blobs_in_bin[bin_index]:
                tasks.append(asyncio.create_task(download_and_decode_glm_file_for_bin(bin_index, blob)))

        results = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []

        # Consume results into ABI raw accumulators and GLM sums
        for r in results:
            if isinstance(r, Exception) or not r:
                continue

            # ABI: (bin_index, short_key, ok, raw_sum, raw_count)
            if len(r) == 5 and r[1] in prod_index:
                bin_index, short_key, ok, raw_sum, raw_count = r
                if not (window_start <= bin_index < window_end):
                    continue
                wi = bin_index - window_start
                if ok and raw_sum is not None and raw_count is not None:
                    s0, c0, k0 = raw_acc[short_key][wi]
                    raw_acc[short_key][wi] = (
                        s0 + raw_sum.astype(np.float32, copy=False),
                        c0 + raw_count.astype(np.int32, copy=False),
                        k0 + 1,
                    )
                    pidx = prod_index[short_key]
                    abi_present_win[wi, pidx] = np.uint8(1)
                    abi_files_ok_window[wi, pidx] = np.int16(min(int(abi_files_ok_window[wi, pidx]) + 1, 32767))

            # GLM: (bin_index, ok, counts)
            else:
                bin_index, ok, counts = r
                if not (window_start <= bin_index < window_end):
                    continue
                wi = bin_index - window_start
                if ok and counts is not None:
                    glm_flash_count_sum[wi] += counts.astype(np.float32, copy=False)
                    glm_bin_present[wi] = np.uint8(1)
                    glm_files_ok[wi] = np.int16(min(int(glm_files_ok[wi]) + 1, 32767))

        # Compute requested scaled file metrics
        abi_actual_valid_files_scaled_win[:, :] = abi_files_ok_window.astype(np.float32) * np.float32(pix_mult)
        with np.errstate(divide="ignore", invalid="ignore"):
            abi_valid_file_fraction_win[:, :] = (
                abi_expected_files_scaled_win / np.where(abi_actual_valid_files_scaled_win > 0, abi_actual_valid_files_scaled_win, np.nan)
            ).astype(np.float32)

        glm_actual_valid_files_scaled_win[:] = glm_files_ok.astype(np.float32) * np.float32(pix_mult)
        with np.errstate(divide="ignore", invalid="ignore"):
            glm_valid_file_fraction_win[:] = (
                glm_expected_files_scaled_win / np.where(glm_actual_valid_files_scaled_win > 0, glm_actual_valid_files_scaled_win, np.nan)
            ).astype(np.float32)

        # Finalize ABI per bin: raw_mean -> coarsen once, valid counts -> coarsen sum
        for wi in range(window_len):
            for key, p_idx in prod_index.items():
                raw_sum, raw_count, ok_files = raw_acc[key][wi]
                if ok_files <= 0:
                    continue

                raw_mean = np.full((raw_y, raw_x), np.nan, dtype=np.float32)
                m = raw_count > 0
                raw_mean[m] = (raw_sum[m] / raw_count[m]).astype(np.float32, copy=False)

                coarse_val = _coarsen_mean_ignore_nan(raw_mean, f)
                coarse_valid_samples = _coarsen_sum_int(raw_count.astype(np.int32, copy=False), f).astype(np.int32, copy=False)

                coarse_val = pad_or_crop_to_shape_2d(coarse_val.astype(np.float32, copy=False), grid_y, grid_x, fill=np.nan)
                coarse_valid_samples = pad_or_crop_to_shape_2d(coarse_valid_samples.astype(np.int32, copy=False), grid_y, grid_x, fill=np.int32(0))

                abi_value_window[wi, p_idx] = coarse_val
                abi_valid_pix_window[wi, p_idx] = coarse_valid_samples

                denom = float(expected_pix_per_output_pixel)
                abi_vpf_window[wi, p_idx] = (coarse_valid_samples.astype(np.float32, copy=False) / denom).astype(np.float32, copy=False)

        # Legacy X tensor uses valid_pixel_fraction per your updated definition
        X_window = np.empty((window_len, NUM_FEATURE_CHANNELS, grid_y, grid_x), dtype=np.float32)
        for wi in range(window_len):
            cmip = abi_value_window[wi, 0]
            acha = abi_value_window[wi, 1]
            tpw  = abi_value_window[wi, 2]

            cmip_vpf = abi_vpf_window[wi, 0]
            acha_vpf = abi_vpf_window[wi, 1]
            tpw_vpf  = abi_vpf_window[wi, 2]

            X_window[wi] = np.stack(
                [
                    cmip,
                    acha,
                    tpw,
                    glm_flash_count_sum[wi],
                    cmip_vpf,
                    acha_vpf,
                    tpw_vpf,
                    ones_grid * np.float32(abi_present_win[wi, 0]),
                    ones_grid * np.float32(abi_present_win[wi, 1]),
                    ones_grid * np.float32(abi_present_win[wi, 2]),
                    ones_grid * np.float32(glm_bin_present[wi]),
                ],
                axis=0,
            )

        t_write0 = time.perf_counter()
        await asyncio.to_thread(
            _write_time_window_to_zarr,
            root,
            window_start,
            X_window,
            abi_value_window,
            abi_expected_pix_window,
            abi_valid_pix_window,
            abi_vpf_window,
            abi_present_win,
            glm_flash_count_sum,
            glm_bin_present,
            glm_files_listed,
            glm_files_ok,
            abi_files_listed_window,
            abi_files_ok_window,
            abi_expected_files_scaled_win,
            abi_actual_valid_files_scaled_win,
            abi_valid_file_fraction_win,
            glm_expected_files_scaled_win,
            glm_actual_valid_files_scaled_win,
            glm_valid_file_fraction_win,
        )
        write_seconds = time.perf_counter() - t_write0
        if write_seconds >= cfg.warn_if_write_slower_than_seconds:
            log(f"writer: WARNING slow write bins {window_start}:{window_end}  {write_seconds:.2f}s")

        log(f"{day_utc.date().isoformat()} wrote bins {window_start:03d}-{window_end-1:03d} / {n_bins-1:03d}")

    log(f"-> wrote merged: {out_zarr_path}  bins={n_bins}  grid={grid_y}x{grid_x}")

    try:
        mark_zarr_store_complete(root)
    except Exception as e:
        log(f"!! WARNING: merged but failed to mark complete: {repr(e)}")
        return False

    return True


# =============================================================================
# Runner
# =============================================================================

def print_config():
    print("-" * 72)
    print("CONFIG")
    print("-" * 72)
    print(json.dumps(cfg.__dict__, indent=2))
    print("-" * 72)


async def run_all_days():
    ensure_dir(cfg.output_root_dir)
    print_config()

    _ = bins_per_day()
    _ = bin_seconds()

    days = list(iter_days_utc(cfg.start_date_utc, cfg.end_date_utc))
    total = len(days)

    connector = aiohttp.TCPConnector(
        limit=cfg.max_concurrent_http_requests,
        limit_per_host=cfg.tcp_limit_per_host,
        ttl_dns_cache=300,
    )
    http_semaphore = asyncio.Semaphore(cfg.max_concurrent_http_requests)

    q: asyncio.Queue[datetime | None] = asyncio.Queue()
    for d in days:
        await q.put(d)
    for _ in range(cfg.max_days_in_flight):
        await q.put(None)

    ok = 0
    fail = 0
    counter_lock = asyncio.Lock()

    async with aiohttp.ClientSession(connector=connector) as session:
        with ProcessPoolExecutor(max_workers=cfg.max_decode_processes) as pool:

            async def worker(worker_id: int):
                nonlocal ok, fail
                while True:
                    day = await q.get()
                    try:
                        if day is None:
                            return
                        good = await process_single_day(day, session, http_semaphore, pool)
                        async with counter_lock:
                            if good:
                                ok += 1
                            else:
                                fail += 1
                    finally:
                        q.task_done()

            tasks = [asyncio.create_task(worker(i)) for i in range(cfg.max_days_in_flight)]
            await q.join()
            await asyncio.gather(*tasks, return_exceptions=True)

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"Days attempted: {total}")
    print(f"Days merged OK: {ok}")
    print(f"Days failed:    {fail}")
    print(f"Output root:    {os.path.abspath(cfg.output_root_dir)}")
    print("=" * 72)


def main():
    try:
        import netCDF4  # noqa
        import pyproj   # noqa
    except Exception:
        print("Missing deps. Install:")
        print("  pip install aiohttp netCDF4 numpy zarr pyproj")
        raise SystemExit(1)

    _ = effective_coarsen_factor()
    _ = bins_per_day()
    _ = bin_seconds()

    asyncio.run(run_all_days())


if __name__ == "__main__":
    main()
