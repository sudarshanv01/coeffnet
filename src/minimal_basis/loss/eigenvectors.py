import itertools

import numpy as np

import torch

from torch.nn import functional as F

from torch import nn


class UnsignedMSELoss(nn.Module):
    def __init__(self):
        super(UnsignedMSELoss, self).__init__()

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
        sign_combinations = torch.tensor(
            list(sign_combinations),
            dtype=torch.float32,
            device=input.device,
            requires_grad=False,
        )
        signed = sign_combinations[:, batch]
        signed = signed.view(-1, signed.shape[-1], 1)
        signed_input = input * signed
        combined_loss = torch.sum((signed_input - target) ** 2, dim=(1, 2))

        loss = torch.min(combined_loss)

        return loss
