"""Parameter count of the two GSSC networks (16.6 million; Methods 2.6).
    uv run --extra gssc python scripts/gssc_param_count.py
"""
from importlib.resources import files

import torch

for n in ('sig_net_v1.pt', 'gru_net_v1.pt'):
    net = torch.load(files('gssc.nets').joinpath(n), weights_only=False, map_location='cpu')
    print(n, type(net).__name__, sum(p.numel() for p in net.parameters()))
