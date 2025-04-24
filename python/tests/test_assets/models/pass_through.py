import torch


class PassThroughNet(torch.nn.Module):
    # x represents our data
    def forward(self, x):
        # Pass data through
        return x
