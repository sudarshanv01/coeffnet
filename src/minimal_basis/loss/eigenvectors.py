import itertools

import torch

from torch import nn


class Unsigned_MSELoss(nn.Module):
    def __init__(self):
        super(Unsigned_MSELoss, self).__init__()

    def forward(self, input, target, batch, batch_size, reduction="sum"):
        """Since eigenvectors are accurate only upto a sign change, we need to
        make sure that the loss is calculated correctly. This function takes
        care of that for multiple graphs in batch."""

        sign_combinations = itertools.product([1, -1], repeat=batch_size)

        losses = []
        for signs in sign_combinations:
            signed = torch.zeros_like(batch)
            signed = signed.reshape(-1, 1)
            for i, sign in enumerate(signs):
                signed[batch == i] = sign
            signed_input = input * signed
            loss = nn.MSELoss(reduction=reduction)(signed_input, target)
            losses.append(loss)
        min_loss = min(losses)
        return min_loss
