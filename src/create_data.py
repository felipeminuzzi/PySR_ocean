import numpy as np
import pandas as pd
import xarray as xr

def get_data(dataset, lat, lon):
    rg_lat      = lat
    rg_long     = lon
    latlon      = dataset.sel(latitude=rg_lat, longitude=rg_long, method='nearest')
    rg_rea      = pd.DataFrame({
        'Time'           : pd.to_datetime(dataset.time.values),
        'Hs'             : latlon.swh.values,
        '10m_direc'      : latlon.dwi.values,
        '10m_speed'      : latlon.wind.values,
        'Peak_period'    : latlon.pp1d.values
    })
    rg_rea      = rg_rea.set_index('Time')
    return rg_rea


path            = './data/raw/era5_reanalysis_utlimos_dados.nc'
data_era        = xr.open_dataset(path)
rg_lat          = -27.24
rg_long         = -47.15
df              = get_data(data_era, rg_lat, rg_long)

df.to_csv('./data/processed/era5_structured_dataset.csv')