import numpy as np
import pandas as pd
import xarray as xr
import config
from typing import Union, Tuple, List, Any

# Define a type alias for the dataset to improve readability
XarrayDataset = xr.Dataset


def single_point_df(dataset: XarrayDataset, lat: float, lon: float) -> pd.DataFrame:
    """
    Extracts time series data for a single point (latitude, longitude)
    and returns a pandas DataFrame.
    """
    # Use method='nearest' for selection and automatically get a new xarray object
    latlon = dataset.sel(latitude=lat, longitude=lon, method='nearest')

    # Create the DataFrame directly from the selected data
    rg_rea = pd.DataFrame({
        'Time': pd.to_datetime(latlon.time.values),
        'Hs': latlon.swh.values,
        '10m_direc': latlon.u10.values,
        '10m_speed': latlon.v10.values,
        'Peak_period': latlon.pp1d.values
    })

    # Set 'Time' as the index
    return rg_rea.set_index('Time')


def space_df(dataset: XarrayDataset, lat: float, lon: float) -> pd.DataFrame:
    """
    Extracts time series data for a single point, calculates several
    non-dimensional parameters, and returns a pandas DataFrame.
    
    The calculations are vectorized for efficiency.
    """
    # Select the nearest data point
    latlon = dataset.sel(latitude=lat, longitude=lon, method='nearest')

    # Create the base DataFrame. Note: use xarray's to_dataframe() for
    # potentially simpler initial creation, but manual construction here
    # is fine too, ensuring 'Time' is included correctly if not the index.
    rg_rea = pd.DataFrame({
        'Time': pd.to_datetime(latlon.time.values),
        'Hs': latlon.swh.values,
        '10m_direc': latlon.u10.values,
        '10m_speed': latlon.v10.values,
        'Peak_period': latlon.pp1d.values,
        'Mean_wave_dir': latlon.mwd.values,
        'Mean_direct_total_swell': latlon.mdts.values,
        'Mean_direct_wind_waves': latlon.mdww.values,
        'Mean_sqr_slope_waves': latlon.msqs.values,
        'Mean_direct_first_swell': latlon.mwd1.values,
        'Mean_direct_second_swell': latlon.mwd2.values,
        'Mean_direct_third_swell': latlon.mwd3.values,
        'latitude': lat,
        'longitude': lon
    })

    # --- Vectorized Calculations for Non-Dimensional Parameters ---
    g = 9.8  # Acceleration due to gravity

    # Calculate wind speed (u10) magnitude
    # Using np.hypot is often cleaner for the hypotenuse
    rg_rea['u10'] = np.hypot(rg_rea['10m_direc'], rg_rea['10m_speed'])

    # Non-dimensional fetch (u10_n)
    # Original formula was (u10**2)/9.8, which is proportional to fetch
    rg_rea['u10_n'] = (rg_rea['u10'] ** 2) / g

    # Non-dimensional peak period (t*)
    rg_rea['Peak_period_n'] = (g * rg_rea['Peak_period']) / rg_rea['u10']

    # Non-dimensional wave height (h*)
    rg_rea['Hs_n'] = rg_rea['Hs'] / rg_rea['u10_n']

    # Wave age (beta)
    rg_rea['Wave_age'] = rg_rea['Peak_period_n'] / (2 * np.pi)

    # 1/t_star, ensuring division by zero is handled (original logic)
    # np.maximum is the vectorized equivalent of max()
    safe_wave_age = np.maximum(rg_rea['Wave_age'], 1e-6)
    rg_rea['1/t_star'] = 1 / safe_wave_age

    # log10(t_star)
    rg_rea['log_tstar'] = np.log10(safe_wave_age)
    
    # log10(y)
    # Note: Assumes 'y' values are non-negative, which they should be
    # given the input variables are squares or physical measurements.
    rg_rea['log_y'] = np.log10(rg_rea['Hs_n'])

    return rg_rea


def main():
    """
    Main execution logic for data processing.
    """
    # --- Configuration and Data Loading ---
    try:
        path = config.raw_df_path
        flag = config.flag

        data_era = xr.open_dataset(path)
    except Exception as e:
        print(f"Error loading data or config: {e}")
        return

    # --- Processing Logic ---
    df: pd.DataFrame
    
    if flag:
        # Process a single point defined in config
        lat = config.latitude
        long = config.longitude
        df = single_point_df(data_era, lat, long)
    else:
        # Process all points in the dataset grid
        lats = data_era.latitude.values
        longs = data_era.longitude.values
        
        # Create a list of (latitude, longitude) tuples for iteration
        # The order in the original code (yv, xv) suggests (lat, long) in the final array
        # np.meshgrid creates coordinate matrices. We want all pairs.
        yv, xv = np.meshgrid(lats, longs)
        # Using a list comprehension is generally more efficient than creating a temporary DataFrame
        lst_latlong: List[Tuple[float, float]] = list(zip(yv.ravel(), xv.ravel()))
        
        # Use a list of DataFrames for efficient concatenation
        all_dfs: List[pd.DataFrame] = []
        
        # Iterating over the list of (lat, long) pairs
        for lat_val, long_val in lst_latlong:
            # Note: space_df is called with (x[1], x[0]) in original code, 
            # which maps to (lat, long) if x is (long, lat). 
            # Here, lst_latlong is (lat, long), so we use (lat_val, long_val).
            aux_df = space_df(data_era, lat_val, long_val)
            all_dfs.append(aux_df)
            
        # Concatenate all DataFrames in one go (much more efficient)
        df = pd.concat(all_dfs, ignore_index=True)
        # Only set the index to 'Time' after concatenation for all points
        df = df.set_index('Time')

    # --- Data Saving ---
    try:
        output_path = './data/processed/era5_structured_dataset.csv'
        df.to_csv(output_path)
        print(f"Successfully saved structured dataset to {output_path}")
    except Exception as e:
        print(f"Error saving data: {e}")

if __name__ == "__main__":
    main()