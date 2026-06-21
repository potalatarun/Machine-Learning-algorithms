# import torch
# import numpy as np
# def pca(data, k) -> torch.Tensor:
#     """
#     Perform PCA on `data`, returning the top `k` principal components as a tensor.
#     Input: Tensor or convertible of shape (n_samples, n_features).
#     Returns: a torch.Tensor of shape (n_features, k), with floats rounded to 4 decimals.
#     Note: If an eigenvector's first non-zero value is negative, flip its sign.
#     """
#     # Your implementation here
#     data_t = torch.tensor(data, dtype=torch.float32)
    
#     means = torch.mean(data_t,dim=0)
#     std = torch.std(data_t, dim=0)

#     data_t = (data_t - means)/std


import torch
import numpy as np

def pca(data, k):

    data_t = torch.tensor(data, dtype=torch.float32)

    # Standardize
    mean = torch.mean(data_t, dim=0)
    std = torch.std(data_t, dim=0, correction=1)

    X = (data_t - mean) / std

    # Covariance matrix
    n_samples = X.shape[0]
    cov = (X.T @ X) / (n_samples - 1)

    # Eigen decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    # Sort descending
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvectors = eigenvectors[:, idx]

    # Top k components
    components = eigenvectors[:, :k]

    # Flip sign if first non-zero entry is negative
    for j in range(components.shape[1]):
        vec = components[:, j]

        for val in vec:
            if abs(val) > 1e-10:
                if val < 0:
                    components[:, j] *= -1
                break

    return torch.round(components, decimals=4)

pca(data = np.array([[1, 2], [3, 4], [5, 6]]), k = 1)
