import torch
import numpy as np

def feature_scaling(data):
    data_t = torch.as_tensor(data, dtype=torch.float32)

    mean = torch.mean(data_t, dim=0)
    std = torch.std(data_t, dim=0, correction=0)

    mini = torch.min(data_t, dim=0).values
    maxi = torch.max(data_t, dim=0).values

    standardized = (data_t - mean) / std
    normalized = (data_t - mini) / (maxi - mini)

    return (
        torch.round(standardized, decimals=4),
        torch.round(normalized, decimals=4)
    )
