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

        loss_idx = torch.argmin(combined_loss)
        signed = signed[loss_idx]
        signed_input = input * signed
        loss = F.mse_loss(signed_input, target, reduction=reduction)

        return loss


class UnsignedL1Loss(nn.Module):
    def __init__(self):
        super(UnsignedL1Loss, self).__init__()

    def forward(self, input, target, batch, batch_size, reduction="sum"):
        """Since eigenvectors are accurate only upto a sign change, we need to
        make sure that the loss is calculated correctly. This function takes
        care of that for multiple graphs in batch."""

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
        combined_loss = torch.sum((signed_input - target).abs(), dim=(1, 2))

        loss_idx = torch.argmin(combined_loss)
        signed = signed[loss_idx]
        signed_input = input * signed
        loss = F.l1_loss(signed_input, target, reduction=reduction)

        return loss


class UnsignedDotProductPreservingMSELoss(nn.Module):
    def __init__(self):
        super(UnsignedDotProductPreservingMSELoss, self).__init__()

    def forward(self, input, target, batch, batch_size, reduction="sum"):
        """No changes are performed on the signs of the predicted quantity. It is
        only supposed to ensure that the \Sum_{j} c_ij @ c_ij.T condition is met, which
        is an alias for taking care of the signs of c_ij."""

        dot_product_input = torch.zeros(batch_size, device=input.device)
        dot_product_target = torch.zeros(batch_size, device=input.device)
        for i in range(batch_size):
            c_ij = input[batch == i]
            c_ij = c_ij.view(-1, 1)
            n_ij = c_ij @ c_ij.T
            dot_product_input[i] = n_ij.sum()

            c_ij = target[batch == i]
            c_ij = c_ij.view(-1, 1)
            n_ij = c_ij @ c_ij.T
            dot_product_target[i] = n_ij.sum()

        loss_dot_product = F.mse_loss(
            dot_product_input, dot_product_target, reduction=reduction
        )
        loss_elements = F.mse_loss(input.abs(), target.abs(), reduction=reduction)

        return loss_dot_product + loss_elements


class UnsignedDotProductPreservingL1Loss(nn.Module):
    def __init__(self):
        super(UnsignedDotProductPreservingL1Loss, self).__init__()

    def forward(self, input, target, batch, batch_size, reduction="sum"):
        """No changes are performed on the signs of the predicted quantity. It is
        only supposed to ensure that the \Sum_{j} c_ij @ c_ij.T condition is met, which
        is an alias for taking care of the signs of c_ij."""

        loss = torch.zeros(batch_size, device=input.device)

        for i in range(batch_size):
            c_ij = input[batch == i]
            c_ij = c_ij.view(-1, 1)
            n_ij_input = c_ij @ c_ij.T

            c_ij = target[batch == i]
            c_ij = c_ij.view(-1, 1)
            n_ij_target = c_ij @ c_ij.T
            l1_nij = F.l1_loss(n_ij_input, n_ij_target, reduction="sum")

            loss[i] = l1_nij

        return loss.sum()
