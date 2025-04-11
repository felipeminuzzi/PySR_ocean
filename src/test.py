import sympy
import numpy as np
from pysr import PySRRegressor
import matplotlib.pyplot as plt

X = 2 * np.random.randn(100, 5)
y = 2.5382 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 0.5

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
ti = np.linspace(0,2*np.pi,100)

plt.plot(ti, y,  'r-', label = 'real')
plt.plot(ti, y2, 'k--', label = 'predicted')
plt.legend()
plt.show()
