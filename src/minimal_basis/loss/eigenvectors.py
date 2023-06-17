import itertools

import numpy as np

import torch

from torch.nn import functional as F

from torch_scatter import scatter_add

from torch import nn


class UnsignedMSELoss(nn.Module):
    def __init__(self, reduction="sum"):
        """Since eigenvectors are accurate only upto a sign change, we need to
        make sure that the loss is calculated correctly. This function takes
        care of that for multiple graphs in batch.

        Args:
            reduction (str, optional): The reduction method. Defaults to "sum".
        """
        self.reduction = reduction
        super(UnsignedMSELoss, self).__init__()

    def forward(self, input, target, batch, batch_size):
        """Forward pass of the loss function.
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
        loss = F.mse_loss(signed_input, target, reduction=self.reduction)

        return loss


class UnsignedL1Loss(nn.Module):
    def __init__(self, reduction="sum"):
        """Since eigenvectors are accurate only upto a sign change, we need to
        make sure that the loss is calculated correctly. This function takes
        care of that for multiple graphs in batch.

        Args:
            reduction (str, optional): The reduction method. Defaults to "sum".
        """
        self.reduction = reduction
        super(UnsignedL1Loss, self).__init__()

    def forward(self, input, target, batch, batch_size):
        """Forward pass of the loss function.
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
        combined_loss = torch.sum((signed_input - target).abs(), dim=(1, 2))

        loss_idx = torch.argmin(combined_loss)
        signed = signed[loss_idx]
        signed_input = input * signed
        loss = F.l1_loss(signed_input, target, reduction=self.reduction)

        return loss


class AbsoluteMSELoss(nn.Module):
    def __init__(self, reduction="sum"):
        """Since eigenvectors are accurate only upto a sign change, this loss function
        takes the absolute value of the predicted eigenvectors and then calculates the
        MSE loss."""
        self.reduction = reduction
        super(AbsoluteMSELoss, self).__init__()

    def forward(self, input, target):
        """Forward pass of the loss function.
        Args:
            input (torch.Tensor): The predicted eigenvectors.
            target (torch.Tensor): The ground truth eigenvectors.
        """
        loss = F.mse_loss(torch.abs(input), torch.abs(target), reduction=self.reduction)
        return loss


class AbsoluteL1Loss(nn.Module):
    def __init__(self, reduction="sum"):
        """Since eigenvectors are accurate only upto a sign change, this loss function
        takes the absolute value of the predicted eigenvectors and then calculates the
        L1 loss."""
        self.reduction = reduction
        super(AbsoluteL1Loss, self).__init__()

    def forward(self, input, target):
        """Forward pass of the loss function.
        Args:
            input (torch.Tensor): The predicted eigenvectors.
            target (torch.Tensor): The ground truth eigenvectors.
        """
        loss = F.l1_loss(torch.abs(input), torch.abs(target), reduction=self.reduction)
        return loss


class RelativeToInitialStateL1Loss(nn.Module):
    def __init__(self, reduction="sum"):
        """The error is calculated relative to the initial state eigenvectors."""
        self.reduction = reduction
        super(RelativeToInitialStateL1Loss, self).__init__()

    def forward(self, input, target, target_initial_state):
        """Forward pass of the loss function.
        Args:
            input (torch.Tensor): The predicted eigenvectors.
            target (torch.Tensor): The ground truth eigenvectors.
            target_initial_state (torch.Tensor): The ground truth initial state eigenvectors.
        """
        absolute_diff_target = target.abs() - target_initial_state.abs()
        absolute_diff_target = absolute_diff_target.abs()
        loss = F.l1_loss(input.abs(), absolute_diff_target, reduction=self.reduction)
        return loss
