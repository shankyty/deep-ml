import torch
import torch.nn.functional as F
from typing import Optional

def to_categorical(x: torch.Tensor, n_col: Optional[int] = None) -> torch.Tensor:
    """
    Perform one-hot encoding on a 1D integer tensor `x`. If `n_col` is not provided, infer it from the max value in `x`.
    """
    # Hint: You can use torch.nn.functional.one_hot
    if n_col is None:
        n_col = int(torch.max(x).item()) + 1
    # one_hot requires a Long tensor (int64), and we cast the result to float 
    # to match the expected [[1., 0., 0.], ...] format from your test cases
    output = torch.zeros((len(x), n_col), dtype=torch.float32)
    output[torch.arange(len(x)), x.to(torch.int64)] = 1.0
    return output
