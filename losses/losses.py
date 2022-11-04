import torch
from torch.nn.functional import cosine_similarity


class MSELoss(torch.nn.Module):

    def __init__(self, flatten: bool = False, reduction: str = 'mean'):
        super().__init__()
        self.loss_fkt = torch.nn.MSELoss(reduction=reduction)
        self.flatten = flatten

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        if self.flatten:
            input = torch.flatten(input, start_dim=1)
            target = torch.flatten(target, start_dim=1)
        loss = self.loss_fkt(input, target)
        return loss


class MAELoss(torch.nn.Module):

    def __init__(self, flatten: bool = False, reduction: str = 'mean'):
        super().__init__()
        self.loss_fkt = torch.nn.L1Loss(reduction=reduction)
        self.flatten = flatten

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        if self.flatten:
            input = torch.flatten(input, start_dim=1)
            target = torch.flatten(target, start_dim=1)
        loss = self.loss_fkt(input, target)
        return loss


class SimilarityLoss(torch.nn.Module):

    def __init__(self, flatten: bool = False, reduction: str = 'mean'):
        super().__init__()
        self.flatten = flatten
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        if self.flatten:
            input = torch.flatten(input, start_dim=1)
            target = torch.flatten(target, start_dim=1)

        loss = -1 * cosine_similarity(input, target, dim=1)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class PoincareLoss(torch.nn.Module):

    def __init__(self, flatten: bool = False, reduction: str = 'mean'):
        super().__init__()
        self.flatten = flatten
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        if self.flatten:
            input = torch.flatten(input, start_dim=1)
            target = torch.flatten(target, start_dim=1)

        # normalize logits
        u = input / torch.norm(input, p=1, dim=-1).unsqueeze(1).cuda()
        # create one-hot encoded target vector
        v = target / torch.norm(target, p=1, dim=-1).unsqueeze(1).cuda()
        # compute squared norms
        u_norm_squared = torch.norm(u, p=2, dim=1)**2
        v_norm_squared = torch.norm(v, p=2, dim=1)**2

        diff_norm_squared = torch.norm(u - v, p=2, dim=1)**2

        # compute delta
        delta = 2 * diff_norm_squared / ((1 - u_norm_squared) *
                                         (1 - v_norm_squared) + 1e-10)
        # compute distance
        loss = torch.arccosh(1 + delta)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
