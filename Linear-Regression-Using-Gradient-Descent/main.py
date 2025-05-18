import numpy as np
def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    """
    Linear regression using gradient descent

    Args:
        X (np.ndarray): Matrix of independent variables
        y (np.ndarray): Vector of dependent variables
        alpha (float): Learning rate
        iterations (int): Number of iterations

    Returns:
        np.ndarray: Vector of coefficients
    """
    m, n = X.shape
    theta = np.zeros((n, 1))  # shape (n, 1) for column vector

    y = y.reshape(-1, 1)  # ensure y is also a column vector

    for _ in range(iterations):
        # Step 1: Predict
        y_pred = X.dot(theta)

        # Step 2: Error
        error = y_pred - y

        # Step 3: Gradient Descent
        gradient = (1 / m) * X.T.dot(error)
        theta = theta - alpha * gradient

    # Return flattened and rounded coefficients
    return np.round(theta.flatten(), 4)



X = np.array([[1, 1], [1, 2], [1, 3]])
y = np.array([1, 2, 3])
alpha = 0.01
iterations = 1000

coefficients = linear_regression_gradient_descent(X, y, alpha, iterations)
print(coefficients) 

