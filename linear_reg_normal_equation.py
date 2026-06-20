import torch

def linear_regression_normal_equation(X, y) -> torch.Tensor:
    """
    Solve linear regression via the normal equation using PyTorch.
    X: Tensor or convertible of shape (m,n); y: shape (m,) or (m,1).
    Returns a 1-D tensor of length n, rounded to 4 decimals.
    """
    X_t = torch.as_tensor(X, dtype=torch.float)
    y_t = torch.as_tensor(y, dtype=torch.float).reshape(-1,1)
    # Your implementation here
    # print(X_t.shape, y_t.shape)
    inv = torch.linalg.pinv(X_t)
    transpose_X = X_t.T
    temp = transpose_X @ X_t
    theta = torch.linalg.pinv(temp) @ transpose_X @ y_t
    return theta
    return torch.linalg.solve(X_t.T, X_t) @ X_t.T @ y
    pass


