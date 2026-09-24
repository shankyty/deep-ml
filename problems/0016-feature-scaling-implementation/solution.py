import torch

def feature_scaling(data) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Standardize and Min-Max normalize input data using PyTorch.
    Input: Tensor or convertible of shape (m,n).
    Returns (standardized_data, normalized_data), both rounded to 4 decimals.
    """
    data_t = torch.as_tensor(data, dtype=torch.float)
    # Your implementation here
    d_min = torch.amin(data_t, dim = 0)
    d_max = torch.amax(data_t, dim = 0)
    d_mean = torch.mean(data_t, dim = 0)
    d_std = torch.std(data_t, dim=0, unbiased=False)
    normalized_data = (data_t - d_min) / (d_max - d_min)
    standardized_data = (data_t - d_mean) / d_std
    normalized_data = torch.round(normalized_data, decimals=4)
    standardized_data = torch.round(standardized_data, decimals=4)
    return standardized_data,normalized_data
