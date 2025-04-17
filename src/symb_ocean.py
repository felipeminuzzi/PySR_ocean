import sympy
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

def plot_results(x,y,leg_name,x_tk,y_tk,fig_name,fig_title):
    
    lst_colors = ['r-','k--','b-','']
    plt.figure()
    plt.title(f'PySR prediction for H_s. Function: {fig_title}')
    for k, i in enumerate(y):
        plt.plot(x, i, lst_colors[k], label = leg_name[k])
    
    plt.ylabel(y_tk)
    plt.xlabel(x_tk)    
    plt.legend()
    plt.savefig(f'./results/pysr_prediction_{fig_name}.png', bbox_inches='tight')

path            = './data/processed/era5_structured_dataset.csv'
df              = pd.read_csv(path)
df['Time']      = pd.to_datetime(df['Time'])
test_set        = df[-(59*24):].copy().reset_index(drop=True)
train_set       = df[-(59*24 + (366*24)):-(59*24)].copy().reset_index(drop=True)

X               = train_set[train_set.columns[2:]].values
y               = train_set[train_set.columns[1]].values

model = PySRRegressor(
    maxsize=20,
    niterations=40,  # < Increase me for better results
    binary_operators=["+", "*"],
    unary_operators=[
        "cos",
        "exp",
        "sin",
        "inv(x) = 1/x",
        # ^ Custom operator (julia syntax)
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    # ^ Custom loss function (julia syntax)
)

model.fit(X, y)
fig_title       = model.latex()
y2              = model.predict(X)
ti              = train_set['Time'].values
mape_model      = mape(y, y2)
print(f'Mean absolute percentage error (MAPE) for train: {mape_model}')
plot_results(ti, [y, y2], ['ERA5 H_s', 'PySR H_S - train'], 'Data', 'Wave height ($H_s$)', 'train-v0', '$'+fig_title +'$')

X_test          = test_set[test_set.columns[2:]].values
y_test          = test_set[test_set.columns[1]].values
y3              = model.predict(X_test)
t_test          = test_set['Time'].values
plot_results(t_test, [y_test, y3], ['ERA5 H_s', 'PySR H_S'], 'Data', 'Wave height ($H_s$)', 'test-v0', '$'+fig_title +'$')

mape_model      = mape(y_test, y3)
error           = erro(y_test, y3)
plot_results(t_test, [error], ['Abs. error'], 'Data', 'Abs. error $\Delta_{\text{rel}}$', 'error-v0', '$'+fig_title +'$' + f'---- MAPE: {mape_model}')


print('###############################################')
print('###############################################')
print('##                                           ##')
print('##      Simulation succesfully finished!     ##')
print('##                                           ##')
print('###############################################')
print('###############################################')