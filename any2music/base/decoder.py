import typing as tp
from torch import nn

class BaseDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: tp.Any):
        raise NotImplementedError()

    def generate(self, inputs: tp.Any):
        raise NotImplementedError()