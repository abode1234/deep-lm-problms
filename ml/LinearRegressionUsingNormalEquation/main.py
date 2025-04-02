import numpy as np

def normal_equation(X, y):
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    theta = np.linalg.inv(X.T @ X) @ X.T @ y

    return [round(val, 4) for val in theta]

X = [[1, 1], [1, 2], [1, 3]]
y = [1, 2, 3]
print(normal_equation(X, y))

