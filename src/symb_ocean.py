import sympy
import os
import config
import glob
import numpy as np
import pandas as pd
from pysr import PySRRegressor
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
    
    return figure_title, mape_model, save_path, t_test, y_test, y3


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

figure_title, mape_model, save_path, t_test, y_test, y3 = predict_future(test_set, model, title_fig)

print(f'Mean absolute percentage error (MAPE) for test: {mape_model}')
plot_results(t_test, [y_test, y3], ['ERA5 H_s', 'PySR H_S'], 'Data', 'Wave height ($H_s$)',
             save_path, 'test', figure_title)

error           = erro(y_test, y3)
plot_results(t_test, [error], ['Rel. error'], 'Data', 'Relative. error $\Delta_{\text{rel}}$',
             save_path, 'error', figure_title)

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