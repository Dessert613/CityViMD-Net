"""Exponential Moving Average utilities."""

from __future__ import annotations

import copy

import torch


class ModelEMA:
    """Maintain a shadow copy of model weights for evaluation."""

    def __init__(self, model, decay=0.9999, device=None):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        self.device = device
        if device is not None:
            self.ema.to(device)
        for param in self.ema.parameters():
            param.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for name, ema_v in self.ema.state_dict().items():
                model_v = msd[name].detach()
                if not torch.is_floating_point(ema_v):
                    ema_v.copy_(model_v)
                else:
                    ema_v.mul_(self.decay).add_(model_v, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, state_dict):
        self.ema.load_state_dict(state_dict)

