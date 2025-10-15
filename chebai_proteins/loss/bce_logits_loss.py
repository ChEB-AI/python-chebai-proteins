import torch


class WrappedBCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    def forward(self, input, target, **kwargs):
        # As the custom passed kwargs are not used in BCEWithLogitsLoss, we can ignore them
        return super().forward(input, target)
