import xarray as xr

def print_dataset_overview(path):

    ds = xr.open_dataset(path, engine="netcdf4", mask_and_scale=True, decode_cf=True)

    print(ds)
    print("\nvars:", list(ds.data_vars))
    print("\ncoords:", list(ds.coords))
    print("\nflag_meanings:", ds.attrs.get("flag_meanings"))
