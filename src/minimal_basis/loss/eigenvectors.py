import itertools

import torch

from torch.nn import functional as F

from torch import nn


class Unsigned_MSELoss(nn.Module):
    def __init__(self):
        super(Unsigned_MSELoss, self).__init__()

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

        sign_combinations = itertools.product([1, -1], repeat=batch_size)
        sign_combinations = list(sign_combinations)
        signs = torch.tensor(sign_combinations, dtype=input.dtype, device=input.device)
        signed = signs[:, batch]
        signed = signed.reshape(-1, signed.shape[-1], 1)
        signed_input = input * signed

        _loss = torch.sum((signed_input - target) ** 2, dim=(1, 2))
        arg_loss = torch.argmin(_loss)

        true_signed_input = input * signed[arg_loss]
        loss = F.mse_loss(true_signed_input, target, reduction=reduction)

        return loss
