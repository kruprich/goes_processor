import xarray as xr

path = '/Users/kevinruprich/Downloads/ABI-L2-ACHAC-2023-010-09-OR_ABI-L2-ACHAC-M6_G16_s20230100901171_e20230100903544_c20230100907054.nc'

ds = xr.open_dataset(path, engine="netcdf4", mask_and_scale=True, decode_cf=True)

print(ds)                  # overview
print(list(ds.data_vars))  # data variables (HT, DQF, etc.)
print(list(ds.coords))     # coords (x, y, etc.)

print(dqf.attrs.get("flag_meanings"))