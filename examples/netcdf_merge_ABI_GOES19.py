"""
GOES ABI (CONUS L2) + GLM (LCFA) -> EXACT 288 5-min bins/day (chunk-windowed, robust)
====================================================================================

CNN / later-transform friendliness added (alongside existing X):
- abi/value[bin, prod, y, x]      float32 (NaN where invalid/missing)
- abi/validfrac[bin, prod, y, x]  float32
- abi/present[bin, prod]          uint8
- glm/flash_count[bin, y, x]      float32
- glm/present[bin]                uint8
- glm/n_files_listed[bin]         int16
- glm/n_files_ok[bin]             int16
- grid/x_rad[x], grid/y_rad[y]    float64 (coarsened grid vectors)
- time/bin_start_ns[bin], time/bin_center_ns[bin]  int64
- abi/prod[prod]                  str

Key semantics retained:
- Always writes ALL 288 bins/day.
- Missing ABI -> value=NaN, validfrac=0, present=0
- Missing GLM -> flash_count=0, glm_present=0
- GLM bin is considered present if ANY GLM file in that bin decodes successfully.

CORE ENFORCEMENT (STRICT 5-MIN WINDOWS):
- For both ABI and GLM, assign a file to bin k (window [t0, t1)) ONLY IF:
    file_start >= t0  AND  file_end < t1
- ABI: select at most one file per bin (earliest start if multiple qualify).
- GLM: include all qualifying files per bin.

Robust GOES filename time parser supports optional fractional digit:
  _sYYYYJJJHHMMSSd  and  _eYYYYJJJHHMMSSd (d optional)

Deps:
  pip install aiohttp netCDF4 numpy zarr pyproj
"""

from __future__ import annotations

import os
import re
import sys
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
# Config
# =============================================================================

@dataclass(frozen=True)
class ProductCfg:
    key: str
    product: str
    must_contain: str
    value_var: str | None = None
    dqf_var: str | None = None


@dataclass(frozen=True)
class Cfg:
    bucket: str = "gcp-public-data-goes-19"

    start_date: str = "2025-10-16"
    end_date: str = "2026-02-01"

    coarsen_factor: int = 4

    # HTTP
    dl_conc: int = 32
    timeout_s: int = 60
    retries: int = 3
    tcp_limit_per_host: int = 96

    # CPU decode
    proc_workers: int = 12

    # Day concurrency
    days_in_flight: int = 3

    # Bin chunking (key principle)
    bin_window: int = 24  # 12, 24, 48 are good

    out_root: str = "./goes_abi_glm"
    overwrite: bool = False

    # Zarr chunking (lag-friendly: chunk across time bins)
    time_chunk: int = 16
    chunk_y: int = 256
    chunk_x: int = 256

    list_conc: int = 48
    warn_write_s: float = 20.0


cfg = Cfg()

PROD_CMIP = ProductCfg(
    key="cmip",
    product="ABI-L2-CMIPC",
    must_contain="ABI-L2-CMIPC-M6C13",
    value_var="CMI",
    dqf_var="DQF",
)

PROD_ACHA = ProductCfg(
    key="acha",
    product="ABI-L2-ACHAC",
    must_contain="ABI-L2-ACHAC-M6",
    value_var="HT",
    dqf_var="DQF",
)

PROD_TPW = ProductCfg(
    key="tpw",
    product="ABI-L2-TPWC",
    must_contain="ABI-L2-TPWC-M6",
    value_var="TPW",
    dqf_var="DQF_Overall",
)

PROD_GLM = ProductCfg(
    key="glm",
    product="GLM-L2-LCFA",
    must_contain="GLM-L2-LCFA",
)

ABI_PRODUCTS = [PROD_CMIP, PROD_ACHA, PROD_TPW]
ABI_PROD_KEYS = ["cmip", "acha", "tpw"]
N_PROD = 3

N_BINS = 288

CHANNEL_NAMES = np.array(
    [
        "cmip_c13_cmi",
        "acha_ht",
        "tpw_tpw",
        "glm_flash_count",
        "cmip_validfrac",
        "acha_validfrac",
        "tpw_validfrac",
        "cmip_present",
        "acha_present",
        "tpw_present",
        "glm_present",
    ],
    dtype="U32",
)
N_CH = int(CHANNEL_NAMES.shape[0])


# =============================================================================
# Logging
# =============================================================================

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def rm_tree(path: str):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def zarr_exists(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, ".zgroup"))


def merged_path(out_root: str, year: int, jday: int) -> str:
    d = os.path.join(out_root, "merged", str(year))
    ensure_dir(d)
    return os.path.join(d, f"{year}{jday:03d}.zarr")


def infer_satellite_from_bucket(bucket: str) -> str:
    """
    Best-effort satellite tag for metadata.
    Examples:
      gcp-public-data-goes-19 -> GOES-19
      gcp-public-data-goes-16 -> GOES-16
    """
    m = re.search(r"goes-(\d+)", bucket)
    if not m:
        return "UNKNOWN"
    return f"GOES-{int(m.group(1))}"


# =============================================================================
# Time helpers
# =============================================================================

# Robust GOES time parsers: support optional fractional digit at end of seconds
# _sYYYYJJJHHMMSS[d] and _eYYYYJJJHHMMSS[d]  (d optional)
_PAT_S = re.compile(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})(\d)?")
_PAT_E = re.compile(r"_e(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})(\d)?")

def _dt_from_parts(y: int, j: int, hh: int, mm: int, ss: int, tenth: int | None) -> datetime:
    base = datetime(y, 1, 1, tzinfo=timezone.utc) + timedelta(days=j - 1, hours=hh, minutes=mm, seconds=ss)
    if tenth is not None:
        base += timedelta(milliseconds=100 * int(tenth))  # tenth-of-second -> 100 ms
    return base

def parse_scan_start_utc(blob_name: str) -> datetime | None:
    m = _PAT_S.search(blob_name)
    if not m:
        return None
    y, j, hh, mm, ss, tenth = m.groups()
    return _dt_from_parts(int(y), int(j), int(hh), int(mm), int(ss), int(tenth) if tenth else None)

def parse_scan_end_utc(blob_name: str) -> datetime | None:
    m = _PAT_E.search(blob_name)
    if not m:
        return None
    y, j, hh, mm, ss, tenth = m.groups()
    return _dt_from_parts(int(y), int(j), int(hh), int(mm), int(ss), int(tenth) if tenth else None)

def date_iter_utc(start_date: str, end_date: str):
    a = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    b = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    cur = a
    while cur <= b:
        yield cur
        cur += timedelta(days=1)

def day0_utc(year: int, jday: int) -> datetime:
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=jday - 1)

def build_5min_bin_starts_ns(year: int, jday: int) -> np.ndarray:
    d0 = day0_utc(year, jday)
    base = np.datetime64(d0.replace(tzinfo=None), "ns")
    return base + (np.arange(N_BINS, dtype=np.int64) * np.timedelta64(5, "m"))


# =============================================================================
# HTTP listing (async JSON API)
# =============================================================================

async def _gcs_list_names_http(
    session: aiohttp.ClientSession,
    bucket: str,
    prefix: str,
    list_sem: asyncio.Semaphore,
) -> list[str]:
    out: list[str] = []
    page_token: str | None = None

    base = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o"
    timeout = aiohttp.ClientTimeout(total=cfg.timeout_s)

    params = {"prefix": prefix, "fields": "items(name),nextPageToken", "maxResults": "1000"}

    while True:
        if page_token:
            params["pageToken"] = page_token
        else:
            params.pop("pageToken", None)

        async with list_sem:
            try:
                async with session.get(base, params=params, timeout=timeout) as r:
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


async def list_day_product_async(
    session: aiohttp.ClientSession,
    prod: ProductCfg,
    year: int,
    jday: int,
    list_sem: asyncio.Semaphore,
) -> list[tuple[datetime, datetime, str]]:
    """
    Return (start_utc, end_utc, blob_name) for all files matching prod.must_contain.
    """
    prefixes = [f"{prod.product}/{year}/{jday:03d}/{hh:02d}/" for hh in range(24)]
    tasks = [asyncio.create_task(_gcs_list_names_http(session, cfg.bucket, pfx, list_sem)) for pfx in prefixes]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    names: list[str] = []
    for res in results:
        if isinstance(res, Exception):
            continue
        names.extend(res)

    triples: list[tuple[datetime, datetime, str]] = []
    for n in names:
        if prod.must_contain not in n:
            continue
        ts = parse_scan_start_utc(n)
        te = parse_scan_end_utc(n)
        if ts is None or te is None:
            continue
        triples.append((ts, te, n))

    triples.sort(key=lambda x: x[0])
    return triples


# =============================================================================
# STRICT 5-min window assignment
# =============================================================================

def abi_select_strict_per_bin(day0: datetime, triples: list[tuple[datetime, datetime, str]]) -> list[str | None]:
    """
    ABI: For each bin [t0, t1), select at most one file with:
        start >= t0 AND end < t1
    If multiple qualify, choose earliest start.
    O(nfiles).
    """
    sel: list[str | None] = [None] * N_BINS
    best_start: list[datetime | None] = [None] * N_BINS

    for ts, te, blob in triples:
        dt0 = (ts - day0).total_seconds()
        k = int(dt0 // 300.0)
        if k < 0 or k >= N_BINS:
            continue

        t0 = day0 + timedelta(seconds=300 * k)
        t1 = t0 + timedelta(seconds=300)

        if ts >= t0 and te < t1:
            bs = best_start[k]
            if bs is None or ts < bs:
                best_start[k] = ts
                sel[k] = blob

    return sel


def glm_group_strict_into_bins(day0: datetime, triples: list[tuple[datetime, datetime, str]]) -> list[list[str]]:
    """
    GLM: For each bin [t0, t1), include file if:
        start >= t0 AND end < t1
    O(nfiles).
    """
    bins: list[list[str]] = [[] for _ in range(N_BINS)]

    for ts, te, blob in triples:
        dt0 = (ts - day0).total_seconds()
        k = int(dt0 // 300.0)
        if k < 0 or k >= N_BINS:
            continue

        t0 = day0 + timedelta(seconds=300 * k)
        t1 = t0 + timedelta(seconds=300)

        if ts >= t0 and te < t1:
            bins[k].append(blob)

    return bins


# =============================================================================
# Downloads
# =============================================================================

async def fetch_bytes(session: aiohttp.ClientSession, blob_name: str) -> bytes | None:
    url = f"https://storage.googleapis.com/{cfg.bucket}/{blob_name}"
    timeout = aiohttp.ClientTimeout(total=cfg.timeout_s)
    for attempt in range(1, cfg.retries + 1):
        try:
            async with session.get(url, timeout=timeout) as r:
                if r.status == 200:
                    return await r.read()
        except Exception:
            pass
        await asyncio.sleep(min(0.25 * attempt, 1.0))
    return None


# =============================================================================
# Shape + netCDF helpers
# =============================================================================

def pad_or_crop_2d(a: np.ndarray, y: int, x: int, *, fill: float) -> np.ndarray:
    if a.shape == (y, x):
        return a
    out = np.full((y, x), fill, dtype=a.dtype)
    yy = min(y, a.shape[0])
    xx = min(x, a.shape[1])
    out[:yy, :xx] = a[:yy, :xx]
    return out


def squeeze_2d(a: np.ndarray, name: str) -> np.ndarray:
    a = np.squeeze(np.asarray(a))
    if a.ndim != 2:
        raise ValueError(f"{name} not 2D after squeeze: shape={a.shape}")
    return a


def squeeze_1d(v: np.ndarray, name: str) -> np.ndarray:
    v = np.squeeze(np.asarray(v))
    if v.ndim != 1:
        raise ValueError(f"{name} not 1D after squeeze: shape={v.shape}")
    return v


# =============================================================================
# ABI probe + decode (reference raw shape, consistent coarsen)
# =============================================================================

def abi_probe_goes_geos_params(payload: bytes) -> dict[str, float | str]:
    """
    Read GOES fixed-grid (GEOS) projection parameters from ABI L2 probe file.
    Expected CF projection variable: goes_imager_projection
    """
    import netCDF4 as nc

    with nc.Dataset("inmem", mode="r", memory=payload) as ds:
        if "goes_imager_projection" not in ds.variables:
            raise KeyError("ABI file missing goes_imager_projection variable")
        p = ds.variables["goes_imager_projection"]

        def _need(name: str):
            if not hasattr(p, name):
                raise KeyError(f"goes_imager_projection missing attribute {name!r}")
            return getattr(p, name)

        return {
            "lon0_deg": float(_need("longitude_of_projection_origin")),
            "h_m": float(_need("perspective_point_height")),
            "a_m": float(_need("semi_major_axis")),
            "b_m": float(_need("semi_minor_axis")),
            "sweep": str(_need("sweep_angle_axis")),
        }


def abi_probe_reference(payload: bytes, value_var: str, dqf_var: str):
    import netCDF4 as nc

    with nc.Dataset("inmem", mode="r", memory=payload) as ds:
        val = squeeze_2d(ds.variables[value_var][:], f"probe:{value_var}").astype(np.float32, copy=False)

        dqf = np.squeeze(np.asarray(ds.variables[dqf_var][:]))
        if dqf.ndim == 0:
            dqf = np.full(val.shape, int(dqf.item()), dtype=np.int16)
        else:
            dqf = squeeze_2d(dqf, f"probe:{dqf_var}").astype(np.int16, copy=False)

        if dqf.shape != val.shape:
            raise ValueError(f"probe dqf shape {dqf.shape} != val shape {val.shape}")

        x_raw = squeeze_1d(ds.variables["x"][:], "probe:x").astype(np.float64, copy=False)
        y_raw = squeeze_1d(ds.variables["y"][:], "probe:y").astype(np.float64, copy=False)

    ref_raw_y, ref_raw_x = int(val.shape[0]), int(val.shape[1])
    goes_geos = abi_probe_goes_geos_params(payload)
    return ref_raw_y, ref_raw_x, x_raw, y_raw, goes_geos


def abi_coarsen_vectors_from_ref(x_raw: np.ndarray, y_raw: np.ndarray, factor: int):
    def trim1(v: np.ndarray):
        n2 = (v.shape[0] // factor) * factor
        return v[:n2]

    x = trim1(x_raw)
    y = trim1(y_raw)
    x_c = x.reshape(x.shape[0] // factor, factor).mean(axis=1)
    y_c = y.reshape(y.shape[0] // factor, factor).mean(axis=1)
    return x_c.astype(np.float64), y_c.astype(np.float64)


def extract_abi_mean_valid_and_frac_refshape(
    payload: bytes,
    value_var: str,
    dqf_var: str,
    factor: int,
    ref_raw_y: int,
    ref_raw_x: int,
):
    import netCDF4 as nc

    with nc.Dataset("inmem", mode="r", memory=payload) as ds:
        val = squeeze_2d(ds.variables[value_var][:], value_var).astype(np.float32, copy=False)

        dqf = np.squeeze(np.asarray(ds.variables[dqf_var][:]))
        if dqf.ndim == 0:
            dqf = np.full(val.shape, int(dqf.item()), dtype=np.int16)
        else:
            dqf = squeeze_2d(dqf, dqf_var).astype(np.int16, copy=False)

        if dqf.shape != val.shape:
            if dqf.size == 1:
                dqf = np.full(val.shape, int(dqf.reshape(-1)[0]), dtype=np.int16)
            else:
                raise ValueError(f"{dqf_var} shape {dqf.shape} != {value_var} shape {val.shape}")

    # Normalize to reference raw shape BEFORE coarsen
    val = pad_or_crop_2d(val, ref_raw_y, ref_raw_x, fill=np.nan)
    dqf = pad_or_crop_2d(dqf, ref_raw_y, ref_raw_x, fill=1).astype(np.int16, copy=False)

    # Consistent trim based on reference raw dims
    y2 = (ref_raw_y // factor) * factor
    x2 = (ref_raw_x // factor) * factor
    val = val[:y2, :x2]
    dqf = dqf[:y2, :x2]

    valid = (dqf == 0)
    y, x = val.shape

    vv = np.where(valid, val, 0.0).astype(np.float32, copy=False)
    vv_sum = vv.reshape(y // factor, factor, x // factor, factor).sum(axis=(1, 3))
    frac = valid.astype(np.float32, copy=False).reshape(y // factor, factor, x // factor, factor).mean(axis=(1, 3))

    cnt = np.maximum(frac * (factor * factor), 1.0).astype(np.float32, copy=False)
    mean_valid = (vv_sum / cnt).astype(np.float32, copy=False)

    # IMPORTANT: if no valid pixels in a block, mean should be NaN
    mean_valid = np.where(frac > 0.0, mean_valid, np.nan).astype(np.float32, copy=False)

    return mean_valid, frac.astype(np.float32, copy=False)


def _decode_abi(payload: bytes, value_var: str, dqf_var: str, factor: int, ref_raw_y: int, ref_raw_x: int):
    return extract_abi_mean_valid_and_frac_refshape(payload, value_var, dqf_var, factor, ref_raw_y, ref_raw_x)


# =============================================================================
# GLM -> ABI grid flash_count
# =============================================================================

def glm_flash_count_on_abi_grid(
    payload: bytes,
    x_vec_rad: np.ndarray,
    y_vec_rad: np.ndarray,
    lon0_deg: float,
    h_m: float,
    a_m: float,
    b_m: float,
    sweep: str,
):
    import netCDF4 as nc
    from pyproj import CRS, Transformer

    with nc.Dataset("inmem", mode="r", memory=payload) as ds:
        if "flash_lat" not in ds.variables or "flash_lon" not in ds.variables or "flash_quality_flag" not in ds.variables:
            return np.zeros((y_vec_rad.shape[0], x_vec_rad.shape[0]), dtype=np.float32)

        lat = np.asarray(ds.variables["flash_lat"][:], dtype=np.float64).reshape(-1)
        lon = np.asarray(ds.variables["flash_lon"][:], dtype=np.float64).reshape(-1)
        q = np.asarray(ds.variables["flash_quality_flag"][:], dtype=np.int16).reshape(-1)

    if lat.size == 0:
        return np.zeros((y_vec_rad.shape[0], x_vec_rad.shape[0]), dtype=np.float32)

    m = (q == 0) & np.isfinite(lat) & np.isfinite(lon)
    if not np.any(m):
        return np.zeros((y_vec_rad.shape[0], x_vec_rad.shape[0]), dtype=np.float32)

    lat = lat[m]
    lon = lon[m]

    geos = CRS.from_proj4(
        f"+proj=geos +lon_0={lon0_deg} +h={h_m} +a={a_m} +b={b_m} +sweep={sweep} +units=m +no_defs"
    )
    wgs84 = CRS.from_epsg(4326)
    tfm = Transformer.from_crs(wgs84, geos, always_xy=True)

    x_m, y_m = tfm.transform(lon, lat)
    x_rad = np.asarray(x_m, dtype=np.float64) / float(h_m)
    y_rad = np.asarray(y_m, dtype=np.float64) / float(h_m)

    xmin, xmax = float(np.min(x_vec_rad)), float(np.max(x_vec_rad))
    ymin, ymax = float(np.min(y_vec_rad)), float(np.max(y_vec_rad))
    inb = (x_rad >= xmin) & (x_rad <= xmax) & (y_rad >= ymin) & (y_rad <= ymax)
    if not np.any(inb):
        return np.zeros((y_vec_rad.shape[0], x_vec_rad.shape[0]), dtype=np.float32)

    x_rad = x_rad[inb]
    y_rad = y_rad[inb]

    # nearest-index assignment
    x_vec = x_vec_rad
    x_asc = x_vec[0] < x_vec[-1]
    if not x_asc:
        x_vec = x_vec[::-1]

    y_vec = y_vec_rad
    y_asc = y_vec[0] < y_vec[-1]
    if not y_asc:
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

    if not x_asc:
        ix = (x_vec_rad.shape[0] - 1) - ix
    if not y_asc:
        iy = (y_vec_rad.shape[0] - 1) - iy

    out = np.zeros((y_vec_rad.shape[0], x_vec_rad.shape[0]), dtype=np.float32)
    np.add.at(out, (iy, ix), 1.0)
    return out


def _decode_glm(payload: bytes, x_vec_rad: np.ndarray, y_vec_rad: np.ndarray, goes_geos: dict[str, float | str]):
    return glm_flash_count_on_abi_grid(
        payload,
        x_vec_rad=x_vec_rad,
        y_vec_rad=y_vec_rad,
        lon0_deg=float(goes_geos["lon0_deg"]),
        h_m=float(goes_geos["h_m"]),
        a_m=float(goes_geos["a_m"]),
        b_m=float(goes_geos["b_m"]),
        sweep=str(goes_geos["sweep"]),
    )


# =============================================================================
# Zarr init (atomic tmp publish)
# =============================================================================

def init_zarr_store_fixed288_atomic(
    out_zarr: str,
    y: int,
    x: int,
    year: int,
    jday: int,
    starts_ns: np.ndarray,
    goes_geos: dict[str, float | str],
):
    if zarr_exists(out_zarr) and not cfg.overwrite:
        return zarr.open_group(out_zarr, mode="r+")

    tmp = out_zarr + ".tmp"
    if os.path.isdir(tmp):
        rm_tree(tmp)

    if zarr_exists(out_zarr) and cfg.overwrite:
        rm_tree(out_zarr)

    compressor = zarr.Blosc(cname="zstd", clevel=1, shuffle=zarr.Blosc.BITSHUFFLE)

    store = zarr.DirectoryStore(tmp)
    root = zarr.group(store=store, overwrite=True)

    root.attrs.update(
        {
            "schema_version": "v2_ml_friendly",
            "year": year,
            "jday": jday,
            "coarsen_factor": cfg.coarsen_factor,
            "glm_aggregation": "STRICT [t0, t0+5min): start>=t0 and end<t1",
            "abi_selection": "STRICT [t0, t0+5min): start>=t0 and end<t1",
            "filled_missing_with_nan": True,
            "channels": int(N_CH),
            "bins_per_day": int(N_BINS),
            "compressor": "blosc:zstd:clevel1:bitshuffle",
            "grid_shape": [int(y), int(x)],
            "time_chunk": int(cfg.time_chunk),
            "abi_products": [p.product for p in ABI_PRODUCTS],
            "abi_value_vars": [p.value_var for p in ABI_PRODUCTS],
            "abi_dqf_vars": [p.dqf_var for p in ABI_PRODUCTS],
            "glm_product": PROD_GLM.product,
            "data_source_bucket": cfg.bucket,
            "data_source_satellite": infer_satellite_from_bucket(cfg.bucket),
            "goes_geos": goes_geos,
        }
    )

    # Coordinates / metadata
    root.create_dataset("channel", shape=(N_CH,), dtype="U32", chunks=(N_CH,), overwrite=True)[:] = CHANNEL_NAMES

    root.create_dataset("time/bin_start_ns", shape=(N_BINS,), chunks=(1024,), dtype="i8", overwrite=True)[:] = (
        starts_ns.astype("datetime64[ns]").astype(np.int64)
    )
    root.create_dataset("time/bin_center_ns", shape=(N_BINS,), chunks=(1024,), dtype="i8", overwrite=True)[:] = (
        (starts_ns + np.timedelta64(150, "s")).astype("datetime64[ns]").astype(np.int64)
    )

    root.create_dataset("abi/prod", shape=(N_PROD,), chunks=(N_PROD,), dtype="U16", overwrite=True)[:] = np.array(
        ABI_PROD_KEYS, dtype="U16"
    )

    root.create_dataset("grid/x_rad", shape=(x,), chunks=(min(x, 4096),), dtype="f8", overwrite=True)
    root.create_dataset("grid/y_rad", shape=(y,), chunks=(min(y, 4096),), dtype="f8", overwrite=True)

    # Original monolithic tensor (retained)
    root.create_dataset(
        "X",
        shape=(N_BINS, N_CH, y, x),
        chunks=(cfg.time_chunk, N_CH, cfg.chunk_y, cfg.chunk_x),
        dtype="f4",
        overwrite=True,
        fill_value=np.nan,
        compressor=compressor,
    )

    # ML-friendly separated tensors
    root.create_dataset(
        "abi/value",
        shape=(N_BINS, N_PROD, y, x),
        chunks=(cfg.time_chunk, N_PROD, cfg.chunk_y, cfg.chunk_x),
        dtype="f4",
        overwrite=True,
        fill_value=np.nan,
        compressor=compressor,
    )
    root.create_dataset(
        "abi/validfrac",
        shape=(N_BINS, N_PROD, y, x),
        chunks=(cfg.time_chunk, N_PROD, cfg.chunk_y, cfg.chunk_x),
        dtype="f4",
        overwrite=True,
        fill_value=0.0,
        compressor=compressor,
    )
    root.create_dataset(
        "abi/present",
        shape=(N_BINS, N_PROD),
        chunks=(N_BINS, N_PROD),
        dtype="u1",
        overwrite=True,
        fill_value=0,
        compressor=compressor,
    )

    root.create_dataset(
        "glm/flash_count",
        shape=(N_BINS, y, x),
        chunks=(cfg.time_chunk, cfg.chunk_y, cfg.chunk_x),
        dtype="f4",
        overwrite=True,
        fill_value=0.0,
        compressor=compressor,
    )
    root.create_dataset(
        "glm/present",
        shape=(N_BINS,),
        chunks=(N_BINS,),
        dtype="u1",
        overwrite=True,
        fill_value=0,
        compressor=compressor,
    )
    root.create_dataset(
        "glm/n_files_listed",
        shape=(N_BINS,),
        chunks=(N_BINS,),
        dtype="i2",
        overwrite=True,
        fill_value=0,
        compressor=compressor,
    )
    root.create_dataset(
        "glm/n_files_ok",
        shape=(N_BINS,),
        chunks=(N_BINS,),
        dtype="i2",
        overwrite=True,
        fill_value=0,
        compressor=compressor,
    )

    if os.path.isdir(out_zarr):
        rm_tree(out_zarr)
    shutil.move(tmp, out_zarr)
    return zarr.open_group(out_zarr, mode="r+")


def _zarr_write_window_sync(
    root,
    k0: int,
    X_blk: np.ndarray,
    abi_value_blk: np.ndarray,
    abi_vf_blk: np.ndarray,
    abi_pres_blk: np.ndarray,
    glm_cnt_blk: np.ndarray,
    glm_pres_blk: np.ndarray,
    glm_listed_blk: np.ndarray,
    glm_ok_blk: np.ndarray,
):
    """
    Writes one contiguous bin window [k0:k0+nwin] into all datasets.
    All inputs are numpy arrays sized to nwin.
    """
    k1 = k0 + X_blk.shape[0]

    root["X"][k0:k1, :, :, :] = X_blk

    root["abi/value"][k0:k1, :, :, :] = abi_value_blk
    root["abi/validfrac"][k0:k1, :, :, :] = abi_vf_blk
    root["abi/present"][k0:k1, :] = abi_pres_blk

    root["glm/flash_count"][k0:k1, :, :] = glm_cnt_blk
    root["glm/present"][k0:k1] = glm_pres_blk
    root["glm/n_files_listed"][k0:k1] = glm_listed_blk
    root["glm/n_files_ok"][k0:k1] = glm_ok_blk


# =============================================================================
# Per-day processing (chunk windows)
# =============================================================================

async def process_one_day(
    day: datetime,
    session: aiohttp.ClientSession,
    http_sem: asyncio.Semaphore,
    pool: ProcessPoolExecutor,
) -> bool:
    year, jday = day.year, int(day.strftime("%j"))
    out_zarr = merged_path(cfg.out_root, year, jday)

    if zarr_exists(out_zarr) and not cfg.overwrite:
        log(f"SKIP {day.date().isoformat()} ({year}{jday:03d}) already merged")
        return True

    log(f"=== {day.date().isoformat()} ({year}{jday:03d}) ABI+GLM chunk-window (STRICT 5-min) ===")

    d0 = day0_utc(year, jday)
    starts_ns = build_5min_bin_starts_ns(year, jday)

    # List all objects up-front
    list_sem = asyncio.Semaphore(cfg.list_conc)
    t_list0 = time.perf_counter()
    log("listing objects (async)...")

    abi_tasks = [asyncio.create_task(list_day_product_async(session, prod, year, jday, list_sem)) for prod in ABI_PRODUCTS]
    glm_task = asyncio.create_task(list_day_product_async(session, PROD_GLM, year, jday, list_sem))

    abi_triples_list = await asyncio.gather(*abi_tasks)
    glm_triples = await glm_task

    log(f"listing done in {time.perf_counter() - t_list0:.2f}s")

    # STRICT 5-min mapping
    abi_sel: dict[str, list[str | None]] = {}
    for prod, triples in zip(ABI_PRODUCTS, abi_triples_list):
        abi_sel[prod.key] = abi_select_strict_per_bin(d0, triples) if triples else [None] * N_BINS

    glm_bins = glm_group_strict_into_bins(d0, glm_triples) if glm_triples else [[] for _ in range(N_BINS)]

    # Probe reference ABI (shape + x/y + goes projection)
    log("probing ABI reference (raw shape + x/y + goes_imager_projection)...")
    probe_payload: bytes | None = None
    probe_prod: ProductCfg | None = None
    probe_blob: str | None = None

    for prod in ABI_PRODUCTS:
        blob = next((b for b in abi_sel[prod.key] if b), None)
        if blob:
            probe_prod = prod
            probe_blob = blob
            async with http_sem:
                probe_payload = await fetch_bytes(session, blob)
            if probe_payload:
                break

    if probe_payload is None or probe_prod is None:
        log("!! no ABI probe downloadable; skipping day")
        return False

    try:
        ref_raw_y, ref_raw_x, x_raw, y_raw, goes_geos = abi_probe_reference(
            probe_payload,
            probe_prod.value_var,  # type: ignore[arg-type]
            probe_prod.dqf_var,    # type: ignore[arg-type]
        )
        x_vec, y_vec = abi_coarsen_vectors_from_ref(x_raw, y_raw, cfg.coarsen_factor)
        y0, x0 = int(y_vec.shape[0]), int(x_vec.shape[0])
        log(
            f"probe OK: ref_raw={ref_raw_y}x{ref_raw_x} -> coarsened={y0}x{x0} "
            f"lon0={goes_geos['lon0_deg']} h={goes_geos['h_m']} sweep={goes_geos['sweep']} "
            f"({os.path.basename(probe_blob or '')})"
        )
    except Exception as e:
        log(f"!! probe failed: {repr(e)}")
        return False

    # Init zarr (store goes_geos from probe)
    try:
        root = init_zarr_store_fixed288_atomic(
            out_zarr,
            y=y0,
            x=x0,
            year=year,
            jday=jday,
            starts_ns=starts_ns,
            goes_geos=goes_geos,
        )
    except Exception as e:
        log(f"!! init zarr failed: {repr(e)}")
        return False

    # Write grid vectors once (coarsened rad coords)
    try:
        root["grid/x_rad"][:] = x_vec.astype(np.float64, copy=False)
        root["grid/y_rad"][:] = y_vec.astype(np.float64, copy=False)
    except Exception:
        pass

    loop = asyncio.get_running_loop()
    write_lock = asyncio.Lock()

    async def dl_decode_abi(bin_k: int, prod: ProductCfg, blob: str):
        async with http_sem:
            payload = await fetch_bytes(session, blob)
        if payload is None:
            return (bin_k, prod.key, False, None, None)

        try:
            v, f = await loop.run_in_executor(
                pool,
                _decode_abi,
                payload,
                prod.value_var,
                prod.dqf_var,
                cfg.coarsen_factor,
                ref_raw_y,
                ref_raw_x,
            )  # type: ignore[arg-type]
            v2 = pad_or_crop_2d(v.astype(np.float32, copy=False), y0, x0, fill=np.nan)
            f2 = pad_or_crop_2d(f.astype(np.float32, copy=False), y0, x0, fill=0.0)
            return (bin_k, prod.key, True, v2, f2)
        except Exception:
            return (bin_k, prod.key, False, None, None)

    async def dl_decode_glm(bin_k: int, blob: str):
        async with http_sem:
            payload = await fetch_bytes(session, blob)
        if payload is None:
            return (bin_k, False, None)

        try:
            cnt = await loop.run_in_executor(pool, _decode_glm, payload, x_vec, y_vec, goes_geos)
            cnt2 = pad_or_crop_2d(cnt.astype(np.float32, copy=False), y0, x0, fill=0.0)
            return (bin_k, True, cnt2)
        except Exception:
            return (bin_k, False, None)

    ones = np.ones((y0, x0), dtype=np.float32)

    for k0 in range(0, N_BINS, cfg.bin_window):
        k1 = min(N_BINS, k0 + cfg.bin_window)
        nwin = k1 - k0

        cm_v = [np.full((y0, x0), np.nan, dtype=np.float32) for _ in range(nwin)]
        ac_v = [np.full((y0, x0), np.nan, dtype=np.float32) for _ in range(nwin)]
        tp_v = [np.full((y0, x0), np.nan, dtype=np.float32) for _ in range(nwin)]

        cm_f = [np.zeros((y0, x0), dtype=np.float32) for _ in range(nwin)]
        ac_f = [np.zeros((y0, x0), dtype=np.float32) for _ in range(nwin)]
        tp_f = [np.zeros((y0, x0), dtype=np.float32) for _ in range(nwin)]

        cm_m = [np.uint8(0) for _ in range(nwin)]
        ac_m = [np.uint8(0) for _ in range(nwin)]
        tp_m = [np.uint8(0) for _ in range(nwin)]

        glm_sum = [np.zeros((y0, x0), dtype=np.float32) for _ in range(nwin)]
        glm_m = [np.uint8(0) for _ in range(nwin)]

        glm_listed = np.zeros((nwin,), dtype=np.int16)
        glm_ok = np.zeros((nwin,), dtype=np.int16)
        for k in range(k0, k1):
            wi = k - k0
            glm_listed[wi] = np.int16(min(len(glm_bins[k]), 32767))

        tasks: list[asyncio.Task] = []
        for k in range(k0, k1):
            for prod in ABI_PRODUCTS:
                blob = abi_sel[prod.key][k]
                if blob is not None:
                    tasks.append(asyncio.create_task(dl_decode_abi(k, prod, blob)))
            for blob in glm_bins[k]:
                tasks.append(asyncio.create_task(dl_decode_glm(k, blob)))

        results = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []

        for r in results:
            if isinstance(r, Exception) or not r:
                continue

            if r[1] in ("cmip", "acha", "tpw"):
                bin_k, pkey, pres, v, f = r
                if not (k0 <= bin_k < k1):
                    continue
                wi = bin_k - k0
                if pres and v is not None and f is not None:
                    if pkey == "cmip":
                        cm_m[wi] = np.uint8(1)
                        cm_v[wi], cm_f[wi] = v, f
                    elif pkey == "acha":
                        ac_m[wi] = np.uint8(1)
                        ac_v[wi], ac_f[wi] = v, f
                    else:
                        tp_m[wi] = np.uint8(1)
                        tp_v[wi], tp_f[wi] = v, f
            else:
                bin_k, pres, cnt = r
                if not (k0 <= bin_k < k1):
                    continue
                wi = bin_k - k0
                if pres and cnt is not None:
                    glm_sum[wi] += cnt
                    glm_m[wi] = np.uint8(1)
                    glm_ok[wi] = np.int16(min(int(glm_ok[wi]) + 1, 32767))

        X_blk = np.empty((nwin, N_CH, y0, x0), dtype=np.float32)
        for wi in range(nwin):
            X_blk[wi] = np.stack(
                [
                    cm_v[wi],
                    ac_v[wi],
                    tp_v[wi],
                    glm_sum[wi],
                    cm_f[wi],
                    ac_f[wi],
                    tp_f[wi],
                    ones * np.float32(cm_m[wi]),
                    ones * np.float32(ac_m[wi]),
                    ones * np.float32(tp_m[wi]),
                    ones * np.float32(glm_m[wi]),
                ],
                axis=0,
            )

        abi_value_blk = np.stack([cm_v, ac_v, tp_v], axis=1).astype(np.float32, copy=False)
        abi_vf_blk = np.stack([cm_f, ac_f, tp_f], axis=1).astype(np.float32, copy=False)
        abi_pres_blk = np.stack([cm_m, ac_m, tp_m], axis=1).astype(np.uint8, copy=False)
        glm_cnt_blk = np.stack(glm_sum, axis=0).astype(np.float32, copy=False)
        glm_pres_blk = np.asarray(glm_m, dtype=np.uint8)

        t_write0 = time.perf_counter()
        async with write_lock:
            await asyncio.to_thread(
                _zarr_write_window_sync,
                root,
                k0,
                X_blk,
                abi_value_blk,
                abi_vf_blk,
                abi_pres_blk,
                glm_cnt_blk,
                glm_pres_blk,
                glm_listed,
                glm_ok,
            )
        dtw = time.perf_counter() - t_write0
        if dtw >= cfg.warn_write_s:
            log(f"writer: WARNING slow write bins {k0}:{k1}  {dtw:.2f}s")

        if k0 == 0 or (k0 // cfg.bin_window) % 2 == 0:
            log(f"{day.date().isoformat()} wrote bins {k0:03d}-{k1-1:03d} / {N_BINS-1:03d}")

    log(f"-> wrote merged: {out_zarr}  bins={N_BINS}  grid={y0}x{x0}")
    return True


# =============================================================================
# Runner
# =============================================================================

def print_cfg():
    print("-" * 72)
    print("CONFIG")
    print("-" * 72)
    print(json.dumps(cfg.__dict__, indent=2))
    print("-" * 72)


async def run_all():
    ensure_dir(cfg.out_root)
    print_cfg()

    days = list(date_iter_utc(cfg.start_date, cfg.end_date))
    total = len(days)

    connector = aiohttp.TCPConnector(
        limit=cfg.dl_conc,
        limit_per_host=cfg.tcp_limit_per_host,
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
    )
    http_sem = asyncio.Semaphore(cfg.dl_conc)

    q: asyncio.Queue[datetime | None] = asyncio.Queue()
    for d in days:
        await q.put(d)
    for _ in range(cfg.days_in_flight):
        await q.put(None)

    ok = 0
    fail = 0
    lock = asyncio.Lock()

    async with aiohttp.ClientSession(connector=connector) as session:
        with ProcessPoolExecutor(max_workers=cfg.proc_workers) as pool:

            async def worker(wid: int):
                nonlocal ok, fail
                while True:
                    day = await q.get()
                    try:
                        if day is None:
                            return
                        good = await process_one_day(day, session, http_sem, pool)
                        async with lock:
                            if good:
                                ok += 1
                            else:
                                fail += 1
                    finally:
                        q.task_done()

            tasks = [asyncio.create_task(worker(i)) for i in range(cfg.days_in_flight)]
            await q.join()
            await asyncio.gather(*tasks, return_exceptions=True)

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"Days attempted: {total}")
    print(f"Days merged OK: {ok}")
    print(f"Days failed:    {fail}")
    print(f"Output root:    {os.path.abspath(cfg.out_root)}")
    print("=" * 72)


def main():
    try:
        import netCDF4  # noqa
        import pyproj   # noqa
    except Exception:
        print("Missing deps. Install:")
        print("  pip install aiohttp netCDF4 numpy zarr pyproj")
        sys.exit(1)

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
