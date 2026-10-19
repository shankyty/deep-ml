import torch
from torch.utils.data import TensorDataset, DataLoader

def batch_iterator(X, y=None, batch_size=64):
    """
    Batch iterator for a dataset represented as PyTorch tensors.

    Args:
        X: Tensor of shape (n_samples, n_features)
        y: Optional Tensor of shape (n_samples,)
        batch_size: Number of samples per batch

    Returns:
        List of Tensor batches (or [X_batch, y_batch] pairs if y is provided)
    """
    # Hint: Use DataLoader with shuffle=False to preserve sample order.
    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    
    if y is not None:
        y_tensor = torch.as_tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
    else:
        dataset = TensorDataset(X_tensor)
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    for batch in loader:
        if len(batch) == 1:
            # y is None: yield just the X_batch tensor
            yield batch[0]
        else:
            # y is provided: yield the [X_batch, y_batch] list
            yield list(batch)

