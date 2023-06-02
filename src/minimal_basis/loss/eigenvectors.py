import itertools

import numpy as np

import torch

from torch.nn import functional as F

from torch import nn


class Unsigned_MSELoss(nn.Module):
    def __init__(self):
        super(Unsigned_MSELoss, self).__init__()

    @staticmethod
    def determine_best_sign(_input, _target, _batch, batch_size):
        """Determine the best sign combination for the input."""
        sign_combinations = itertools.product([1, -1], repeat=batch_size)
        sign_combinations = np.array(list(sign_combinations))
        signed = sign_combinations[:, _batch]

        signed = signed.reshape(-1, signed.shape[-1], 1)
        signed_input = _input * signed

        combined_loss = np.sum((signed_input - _target) ** 2, axis=(1, 2))
        arg_loss = np.argmin(combined_loss)

        return signed[arg_loss]

    def forward(self, input, target, batch, batch_size, reduction="sum"):
        """Since eigenvectors are accurate only upto a sign change, we need to
        make sure that the loss is calculated correctly. This function takes
        care of that for multiple graphs in batch.

        Args:
            input (torch.Tensor): The predicted eigenvectors.
            target (torch.Tensor): The ground truth eigenvectors.
            batch (int): The index of the graph in the batch.
            batch_size (int): The size of the batch.
        """
        _input = input.detach().cpu().numpy()
        _target = target.detach().cpu().numpy()
        _batch = batch.detach().cpu().numpy()

        signs = self.determine_best_sign(_input, _target, _batch, batch_size)
        signs = torch.tensor(
            signs, dtype=torch.float32, device=input.device, requires_grad=False
        )
        true_signed_input = input * signs
        loss = F.mse_loss(true_signed_input, target, reduction=reduction)

        return loss
