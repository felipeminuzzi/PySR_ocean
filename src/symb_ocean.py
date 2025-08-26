import sympy
import os
import config
import glob
import numpy as np
import pandas as pd
from pysr import PySRRegressor
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (18,4)

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)/y_true))*100

def erro(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.abs((y_true - y_pred)/y_true)*100

def plot_results(x,y,leg_name,x_tk,y_tk,fig_name,type,fig_title):
    
    lst_colors = ['r-','k--','b-','-g']
    plt.figure()
    plt.title(f'PySR prediction for H_s. Function: {fig_title}')
    for k, i in enumerate(y):
        plt.plot(x, i, lst_colors[k], label = leg_name[k])
    
    plt.ylabel(y_tk)
    plt.xlabel(x_tk)    
    plt.legend()
    plt.savefig(f'{fig_name}pysr_prediction_{type}.png', bbox_inches='tight')
    plt.close()

def plot_region(df,fig_name, type='test'):
    
    n_lat       = len(df['latitude'].unique())
    n_long      = len(df['longitude'].unique())
    dados_pysr  = df['pysr'].values.reshape(n_lat, n_long)
    dados_era   = df['real'].values.reshape(n_lat, n_long)
    error       = df['error'].values.reshape(n_lat, n_long)
    lati        = df['latitude'].unique()
    long        = df['longitude'].unique()
    
    plt.figure()
    plt.subplot(131)
    plt.title('PySR')
    plot2map(long,lati,dados_pysr)
    plt.subplot(132)
    plt.title('ERA5')
    plot2map(long,lati,dados_era)
    plt.subplot(133)
    plt.title('Relative. error $\Delta_{rel}$')
    plot2map(long,lati,error)    
    plt.savefig(f'{fig_name}pysr_prediction_{type}.png', bbox_inches='tight')
    plt.close()

def plot2map(lon, lat, dados):
    map = Basemap(projection='cyl', llcrnrlon=lon.min(), 
                  llcrnrlat=lat.min(), urcrnrlon=lon.max(), 
                  urcrnrlat=lat.max(), resolution='h')
    
    map.fillcontinents(color=(0.55, 0.55, 0.55))
    map.drawcoastlines(color=(0.3, 0.3, 0.3))
    map.drawstates(color=(0.3, 0.3, 0.3))
    map.drawparallels(np.linspace(lat.min(), lat.max(), 6), labels=[1,0,0,0],
                      rotation=90, dashes=[1, 2], color=(0.3, 0.3, 0.3))
    map.drawmeridians(np.linspace(lon.min(), lon.max(), 6), labels=[0,0,0,1], 
                      dashes=[1, 2], color=(0.3, 0.3, 0.3))
    llons, llats = np.meshgrid(lon, lat)
    lons, lats = map(lon, lat)
    m = map.contourf(llons, llats, dados)
    plt.colorbar(m)

    return m

def format_path(path):
    """""Formats the path string in order to avoid conflicts."""

    if path[-1]!='/':
        path = path + '/'

    if not os.path.exists(path):
        os.makedirs(path)

    return path

def predict_future(test, model, fig_title):
    if config.flag:
        X_test          = test[test.columns[2:]]
        y_test          = test[test.columns[1]]
        y3              = model.predict(X_test)
        t_test          = test['Time'].values
        mape_model      = mape(y_test, y3)
        figure_title    = '$'+fig_title +'$' + f'---- Test MAPE: {mape_model}' 
        save_name       =  f'test-v4_nit300' 
        save_path       = format_path(f'./results/{save_name}')
        df_save         = pd.DataFrame({'time': t_test, 'real': y_test, 'pysr': y3}).reset_index(drop=True)
        df_save.to_csv(save_path + 'df_results.csv')
    else:
        test_df         = test[(test['x3'] == config.lat_tst) & 
                                (test['x4'] == config.long_tst)]
        X_test          = test_df[test.columns[2:]]
        y_test          = test_df[test.columns[1]]
        y3              = model.predict(X_test)
        t_test          = test_df['Time'].values  
        mape_model      = mape(y_test, y3)
        figure_title    = '$'+fig_title +'$' + f' -- lat: {config.lat_tst}; long: {config.long_tst}' +f'---- Test MAPE: {mape_model}'  
        save_name       =  f'v6_nit200_lat{config.lat_tst}_long{config.long_tst}'
        save_path       = format_path(f'./results/{save_name}')
        df_save         = pd.DataFrame({'time': t_test, 'real': y_test, 'pysr': y3}).reset_index(drop=True)
        df_save.to_csv(save_path + 'df_results.csv')

    print(f'Mean absolute percentage error (MAPE) for test: {mape_model}')
    plot_results(t_test, [y_test, y3], ['ERA5 H_s', 'PySR H_S'], 'Data', 'Wave height ($H_s$)',
                save_path, 'test', figure_title)

    error              = erro(y_test, y3)
    plot_results(t_test, [error], ['Rel. error'], 'Data', 'Relative. error $\Delta_{\text{rel}}$',
                save_path, 'error', figure_title)

def region_predict(test, model, fig_title):

    df_test     = test[test['Time'] == pd.to_datetime(config.region_time)]
    y_test      = df_test[df_test.columns[1]].values
    X           = df_test[df_test.columns[2:]]
    y3          = model.predict(X)
    mape_model  = mape(y_test, y3)
    figure_title= '$'+fig_title +'$'
    save_name   =  f'v7_nit200_region_date{config.region_time}'
    save_path   = format_path(f'./results/{save_name}')
    df_save     = pd.DataFrame({'real': y_test, 'pysr': y3, 'latitude': df_test[df_test.columns[5]].values, 
                                'longitude': df_test[df_test.columns[6]]}).reset_index(drop=True)
    df_save.to_csv(save_path + 'df_results.csv')    
    df_save['error'] = df_save.apply(lambda row: erro(row['real'], row['pysr']), axis=1)    
    print(f'Mean absolute percentage error (MAPE) for test: {mape_model}')
    plot_region(df_save,save_path)

path            = './data/processed/era5_structured_dataset.csv'
df              = pd.read_csv(path)
df['Time']      = pd.to_datetime(df['Time'])
column_names    = df.columns.tolist()[2:]
var_names       = {}

for k in range(len(column_names)):
    var_names[column_names[k]] = f'x{k}'

df.rename(columns = var_names, inplace=True)
tst_date        = config.test_initial_date
trn_date        = config.train_initial_date

test_set        = df[df['Time'] >= pd.to_datetime(tst_date)].copy().reset_index(drop=True)
train_set       = df[(df['Time'] < pd.to_datetime(tst_date)) & 
                     (df['Time'] >= pd.to_datetime(trn_date))].copy().reset_index(drop=True)
X               = train_set[train_set.columns[2:]]
y               = train_set[train_set.columns[1]]

if config.new_train:
    model = PySRRegressor(
        maxsize=30,
        niterations=200,  # < Increase me for better results
        binary_operators=["*", "+", "-", "/"],
        unary_operators=[
            "cos",
            "exp",
            "sin",
            "inv(x) = 1/x",
            "tan",
            "sqrt",
            "square",
            "log",
            "cube"
        ],
        # early_stop_condition=(
        #     "stop_if(loss) = loss < 1e-4"
        # ),
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        procs = 16,
        batching=True,
        batch_size=1000
    )

    model.fit(X, y)
else:
    model = PySRRegressor.from_file(run_directory=config.model_saved)

title_fig       = model.latex()
y2              = model.predict(X)
ti              = train_set['Time'].values     
if config.flag:
    predict_future(test_set, model, title_fig)
else:
    region_predict(test_set, model, title_fig)

print('###############################################')
print('Legend of variables:                           ')
print(var_names)
print('###############################################')

print('###############################################')
print('Equation:                                      ')
print(title_fig)
print('###############################################')

print('###############################################')
print('###############################################')
print('##                                           ##')
print('##      Simulation succesfully finished!     ##')
print('##                                           ##')
print('###############################################')
print('###############################################')