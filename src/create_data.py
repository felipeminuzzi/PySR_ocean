import numpy as np
import pandas as pd
import xarray as xr
import config

def single_point_df(dataset, lat, lon):
    rg_lat      = lat
    rg_long     = lon
    latlon      = dataset.sel(latitude=rg_lat, longitude=rg_long, method='nearest')
    rg_rea      = pd.DataFrame({
        'Time'           : pd.to_datetime(dataset.time.values),
        'Hs'             : latlon.swh.values,
        '10m_direc'      : latlon.u10.values,
        '10m_speed'      : latlon.v10.values,
        'Peak_period'    : latlon.pp1d.values
    })
    rg_rea      = rg_rea.set_index('Time')
    return rg_rea

def space_df(dataset, lat, lon):
    rg_lat      = lat
    rg_long     = lon
    latlon      = dataset.sel(latitude=rg_lat, longitude=rg_long, method='nearest')
    rg_rea      = pd.DataFrame({
        'Time'           : pd.to_datetime(dataset.time.values),
        'Hs'             : latlon.swh.values,
        '10m_direc'      : latlon.u10.values,
        '10m_speed'      : latlon.v10.values,
        'Peak_period'    : latlon.pp1d.values,
        'latitude'       : rg_lat,
        'longitude'      : rg_long
    })
    #rg_rea      = rg_rea.set_index('Time')
    rg_rea['u10']      = rg_rea.apply(lambda x: (x['10m_direc']**2 + x['10m_speed']**2)**(1/2), axis=1)
    rg_rea['u10_n']    = (rg_rea['u10']**2)/9.8
    rg_rea['Peak_period_n'] = (9.8*rg_rea['Peak_period'])/rg_rea['u10']
    rg_rea['Hs_n']     = rg_rea['Hs']/rg_rea['u10_n']
    rg_rea['Wave_age'] = rg_rea['Peak_period_n']/(2*np.pi)
    rg_rea['1/t_star'] = rg_rea.apply(lambda x: 1/max(x['Wave_age'], 1e-6),axis =1)
    rg_rea['log_tstar']= rg_rea.apply(lambda x: np.log10(max(x['Wave_age'], 1e-6)),axis =1)
    rg_rea['y']        = rg_rea['Hs']/rg_rea['u10_n']
    
    return rg_rea


path            = config.raw_df_path
flag            = config.flag
data_era        = xr.open_dataset(path)

if flag:
    lat         = config.latitude
    long        = config.longitude
    df          = single_point_df(data_era, lat, long)
else:
    lats        = data_era.latitude.values
    longs       = data_era.longitude.values
    yv, xv      = np.meshgrid(lats, longs)
    df_latlong  = pd.DataFrame(dict(long=xv.ravel(), lat=yv.ravel()))
    lst_latlong = df_latlong.values    
    df          = pd.DataFrame()
    for x in lst_latlong:
        aux_df  = space_df(data_era, x[1], x[0])
        df      = pd.concat([df,aux_df], ignore_index = True)
    df          = df.set_index('Time')

df.to_csv('./data/processed/era5_structured_dataset.csv')