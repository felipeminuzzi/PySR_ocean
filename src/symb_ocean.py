import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from pysr import PySRRegressor
import gc # <-- GARBAGE COLLECTOR
import time

# Set up matplotlib configuration
plt.rcParams["figure.figsize"] = (18, 4)


# --- Metrics Functions ---

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-6) -> float:
    """
    Calculates the Mean Absolute Percentage Error (MAPE), avoiding division by zero.

    When a true value (y_true) is zero, it is replaced by a small positive 
    number (epsilon) to prevent division by zero errors.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Create a mask for where y_true is zero
    zero_mask = (y_true == 0)
    
    # Create a copy of y_true to modify
    y_true_safe = np.where(zero_mask, epsilon, y_true)

    return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100


def calculate_relative_error(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Calculates the element-wise relative error in percentage, avoiding division by zero.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Create a safe denominator
    y_true_safe = np.where(y_true == 0, epsilon, y_true)
    
    return np.abs((y_true - y_pred) / y_true_safe) * 100


# --- Plotting Functions ---

def plot_time_series(x: np.ndarray, y_data: list, legend_names: list, x_label: str, y_label: str,
                     save_path: str, plot_type: str, title: str):
    """
    Generates and saves a time series plot comparing real data and predictions,
    or a plot of the relative error.
    """
    lst_colors = ['r-', 'k--', 'b-', '-g']
    plt.figure()
    plt.title(f'PySR prediction for H_s. Function: {title}')

    for k, i in enumerate(y_data):
        plt.plot(x, i, lst_colors[k % len(lst_colors)], label=legend_names[k])

    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend()
    plt.savefig(f'{save_path}pysr_prediction_{plot_type}.png', bbox_inches='tight')
    plt.close()


def plot_geographic_region(df: pd.DataFrame, save_path: str, mape_value: float, plot_type: str = 'test'):
    """
    Generates and saves a 3-panel plot for geographical data: PySR prediction, ERA5 data,
    and relative error.
    """
    
    n_lat = len(df['latitude'].unique())
    n_long = len(df['longitude'].unique())

    dados_pysr = df['pysr'].values.reshape(n_lat, n_long)
    dados_era = df['real'].values.reshape(n_lat, n_long)
    error = df['error'].values.reshape(n_lat, n_long)

    lati = df['latitude'].unique()
    long = df['longitude'].unique()

    cb_min = min(dados_era.min(), dados_pysr.min())
    cb_max = max(dados_era.max(), dados_pysr.max())

    plt.figure()
    
    # PySR Plot
    plt.subplot(131)
    plt.title('PySR')
    _plot_on_map(long, lati, dados_pysr, cb_min, cb_max, red=False)
    
    # ERA5 Plot
    plt.subplot(132)
    plt.title('ERA5')
    _plot_on_map(long, lati, dados_era, cb_min, cb_max, red=False)
    
    # Error Plot
    plt.subplot(133)
    plt.title('$\Delta_{rel}$' + f' -- MAPE: {round(mape_value, 2)}')
    _plot_on_map(long, lati, error, cb_min, cb_max, red=True)
    
    plt.savefig(f'{save_path}pysr_prediction_{plot_type}.png', bbox_inches='tight')
    plt.close()


def _plot_on_map(lon: np.ndarray, lat: np.ndarray, data: np.ndarray, v_min: float, v_max: float, red: bool):
    """
    Internal helper function to plot data using Basemap.
    """
    map_proj = Basemap(projection='cyl', llcrnrlon=lon.min(), 
                       llcrnrlat=lat.min(), urcrnrlon=lon.max(), 
                       urcrnrlat=lat.max(), resolution='h')
    
    map_proj.fillcontinents(color=(0.55, 0.55, 0.55))
    map_proj.drawcoastlines(color=(0.3, 0.3, 0.3))
    map_proj.drawstates(color=(0.3, 0.3, 0.3))
    
    # Draw parallels and meridians
    map_proj.drawparallels(np.linspace(lat.min(), lat.max(), 6), labels=[1, 0, 0, 0],
                          rotation=90, dashes=[1, 2], color=(0.3, 0.3, 0.3))
    map_proj.drawmeridians(np.linspace(lon.min(), lon.max(), 6), labels=[0, 0, 0, 1], 
                          dashes=[1, 2], color=(0.3, 0.3, 0.3))
    
    llats, llons = np.meshgrid(lat, lon)
    
    levels = np.linspace(math.floor(v_min), math.ceil(v_max), 10)
    
    if red:
        m = map_proj.contourf(llons, llats, data, cmap='Reds')
    else:
        m = map_proj.contourf(llons, llats, data, levels=levels)
    
    plt.colorbar(m)
    return m


# --- Utility Functions ---

def format_and_create_path(path: str) -> str:
    """
    Formats the path string and creates the directory if it doesn't exist.
    Ensures the path ends with a '/'.
    """
    if not path.endswith('/'):
        path = path + '/'

    os.makedirs(path, exist_ok=True)
    return path


# --- Prediction/Execution Functions ---

def time_series_predict(test_data: pd.DataFrame, model: PySRRegressor, fig_title: str):
    """
    Performs prediction for a time series (either a single location or aggregated)
    and plots the results.
    """
    if config.flag:
        # Predict all data in test_data (aggregated or entire test set)
        X_test = test_data.iloc[:, 2:]
        y_test = test_data.iloc[:, 1]
        save_name = 'test-v4_nit300'
    else:
        # Predict for a specific lat/long
        test_df_location = test_data[(test_data['x3'] == config.lat_tst) & 
                                     (test_data['x4'] == config.long_tst)]
        X_test = test_df_location.iloc[:, 2:]
        y_test = test_df_location.iloc[:, 1]
        test_data = test_df_location # Use filtered data for subsequent steps
        save_name = f'v6_nit200_lat{config.lat_tst}_long{config.long_tst}'

    y_pred = model.predict(X_test)
    t_test = test_data['Time'].values
    mape_model = calculate_mape(y_test, y_pred)
    relative_error = calculate_relative_error(y_test, y_pred)
    
    # Save results
    save_path = format_and_create_path(f'./results/{save_name}')
    df_save = pd.DataFrame({'time': t_test, 'real': y_test, 'pysr': y_pred}).reset_index(drop=True)
    df_save.to_csv(save_path + 'df_results.csv')

    print(f'Mean absolute percentage error (MAPE) for test: {mape_model:.2f}%')
    
    figure_title = f'${fig_title}$' + f'---- Test MAPE: {mape_model:.2f}' 
    if not config.flag:
        figure_title += f' -- lat: {config.lat_tst}; long: {config.long_tst}'

    # Plot results
    plot_time_series(t_test, [y_test, y_pred], ['ERA5 H_s', 'PySR H_S'], 'Data', 
                     'Wave height ($H_s$)', save_path, 'test', figure_title)

    # Plot relative error
    plot_time_series(t_test, [relative_error], ['Rel. error'], 'Data', 
                     'Relative. error $\Delta_{\\text{rel}}$', save_path, 'error', figure_title)


def region_predict(test_data: pd.DataFrame, model: PySRRegressor, fig_title: str, var_names: dict):
    """
    Performs prediction for a specific geographical region at a single time step
    and plots the results on a map.
    """
    # Filter for the specific time step
    df_test_region = test_data[test_data['Time'] == pd.to_datetime(config.region_time)]
    y_test = df_test_region[config.target_var].values
    
    # Prepare features for prediction
    feature_columns = [v for k, v in var_names.items()]
    X = df_test_region[feature_columns]
    
    y_pred = model.predict(X)

    U_MIN = getattr(config, 'u10_min', 2.0)
    mask  = df_test_region['u10'].values >= config.u_min if 'u10' in df_test_region.columns else np.ones(len(df_test_region), dtype=bool)

    y3_masked, y_test_mask = y_pred.copy(), y_test.copy()
    y3_masked[~mask], y_test_mask[~mask] = np.nan, np.nan
    mape_model = calculate_mape(y_test, y_pred)
    
    # Prepare DataFrame for saving and plotting
    df_save = pd.DataFrame({
        'real': y_test.flatten(), 
        'pysr': y_pred, 
        'latitude': df_test_region[var_names['latitude']].values, 
        'longitude': df_test_region[var_names['longitude']].values
    }).reset_index(drop=True)
    
    # Calculate relative error and append to the DataFrame
    df_save['error'] = df_save.apply(lambda row: calculate_relative_error(row['real'], row['pysr']), axis=1)    
    
    # Save results
    name_model = config.model_saved.split('/')[-1][-6:]
    save_name = f'v11_nit300_region_date{config.region_time}_{name_model}'
    save_path = format_and_create_path(f'./results/{save_name}')
    df_save.to_csv(save_path + 'df_results.csv')    
    
    print(f'Mean absolute percentage error (MAPE) for test: {mape_model:.2f}%')

    # Plot results on map
    plot_geographic_region(df_save, save_path, mape_model)


# --- Main Execution ---

def main():
    """
    Main function to load data, prepare sets, train/load the PySR model,
    and perform predictions.v_names
    """
    
    # 1. Data Loading and Preparation
    path = './data/processed/era5_structured_dataset.csv'
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {path}. Exiting.")
        return
        
    df['Time'] = pd.to_datetime(df['Time'])

    # Select and order necessary columns: Time, Target, Features
    required_cols = ['Time'] + config.target_var + config.feature_var 
    df = df[required_cols]

    # Create mapping for symbolic regression variables (x0, x1, ...)
    column_names = config.feature_var
    var_names = {column_names[k]: f'x{k}' for k in range(len(column_names))}
    df.rename(columns=var_names, inplace=True)

    # Split data into train and test sets
    tst_date = pd.to_datetime(config.test_initial_date)
    trn_date = pd.to_datetime(config.train_initial_date)

    test_set = df[df['Time'] >= tst_date].copy().reset_index(drop=True)
    train_set = df[(df['Time'] < tst_date) & 
                   (df['Time'] >= trn_date)].copy().reset_index(drop=True)

    # --- MEMORY OPTIMIZATION STEP 1: Release original DataFrame (df) ---
    # After test_set and train_set are created, the original df is no longer needed.
    del df
    gc.collect() 
    print("Released memory for the main 'df' variable.")
    # --- END MEMORY RELEASE ---

    # --- FILTERING LOGIC: Apply u10 filter ONLY to train_set ---
    u10_var_name = var_names.get('u10')
    if u10_var_name and hasattr(config, 'u_min'):
        initial_rows = len(train_set)
        # Filter: Keep rows where u10 is >= u_min
        train_set = train_set[train_set[u10_var_name] >= config.u_min].copy()
        dropped_rows = initial_rows - len(train_set)
        print(f"Train Set Filtered: Dropped {dropped_rows} rows where u10 < {config.u_min}.")
    elif u10_var_name:
        print("Warning: 'u10' is a feature but 'config.u_min' is not defined. No filtering applied to train set.")
    # --- END FILTERING LOGIC ---    

    # Prepare input (X) and output (y) for training
    X = train_set[[i for i in var_names.values()]] 
    y = train_set[config.target_var]

    # Convert X and y to the required NumPy format with smaller dtype for PySR
    X = X.values.astype(np.float32)
    y = y.values.astype(np.float32)
    
    # --- MEMORY OPTIMIZATION STEP 2: Release train_set DataFrame ---
    # train_set is now superseded by the NumPy arrays X and y.
    del train_set
    gc.collect() 
    print("Released memory for the 'train_set' variable.")
    # --- END MEMORY RELEASE ---

    # 2. Model Training or Loading
    if config.new_train:
        model = PySRRegressor(
            maxsize=30,
            niterations=300,
            binary_operators=["*", "+", "-", "/"],
            unary_operators=["exp", "inv(x) = 1/x", "sqrt", "square", "log"],
            extra_sympy_mappings={"inv": lambda x: 1 / x},
            elementwise_loss="loss(prediction, target) = (prediction - target)^2",
            procs=16,
            batching=True,
            batch_size=1000,
            ncycles_per_iteration=500
        )
        print("Starting new PySR training...")
        
        # --- TRAINING TIME MEASUREMENT ---
        start_time = time.time()
        model.fit(X, y)
        end_time = time.time()
        training_time = end_time - start_time
        # --- END TIME MEASUREMENT ---
        
        print(f"PySR training completed in: {training_time:.2f} seconds.")
    else:
        print(f"Loading model from {config.model_saved}...")
        try:
            model = PySRRegressor.from_file(run_directory=config.model_saved)
        except FileNotFoundError:
            print(f"Error: Saved model not found at {config.model_saved}. Check 'config.model_saved'. Exiting.")
            return

    # 3. Prediction and Output
    title_fig = model.latex()
    
    # Call the appropriate prediction function based on config.flag
    if config.flag:
        time_series_predict(test_set, model, title_fig)
    else:
        region_predict(test_set, model, title_fig, var_names)

    # Final Output Summary
    print('\n' + '#'*47)
    print('Legend of variables:                           ')
    print(var_names)
    print('#'*47)

    print('\n' + '#'*47)
    print('Equation:                                      ')
    print(title_fig)
    print('#'*47)

    print('\n' + '#'*47)
    print('##                                           ##')
    print('##      Simulation succesfully finished!     ##')
    print('##                                           ##')
    print('#'*47)


if __name__ == '__main__':

    import config 
    main()