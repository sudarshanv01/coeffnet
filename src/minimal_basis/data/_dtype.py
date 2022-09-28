import torch

# Staying consistent with Mingjian's datatypes.
# See here: https://github.com/mjwen/eigenn/blob/main/eigenn/data/_dtype.py

DTYPE = torch.get_default_dtype()
DTYPE_INT = torch.int64
DTYPE_BOOL = torch.bool
TORCH_FLOATS = (torch.float16, torch.float32, torch.float64)
TORCH_INTS = (torch.int16, torch.int32, torch.int64)
