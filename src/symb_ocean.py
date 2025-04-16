import sympy
import numpy as np
import pandas as pd
from pysr import PySRRegressor
import matplotlib.pyplot as plt

def plot_results(x,y1,y2, x_name, y_name):
    plt.figure()
    plt.plot(x, y1,  'r-', label = 'ERA5 H_s')
    plt.plot(x, y2, 'k--', label = 'PySR H_s')
    plt.ylabel(x_name)
    plt.xlabel(y_name)    
    plt.legend()
    plt.show()

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

print(model)

y2 = model.predict(X)
ti = train_set['Time'].values

plot_results(ti, y, y2, 'Data', 'Wave height (Hs)')
breakpoint()