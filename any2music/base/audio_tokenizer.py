from abc import ABC, abstractmethod
import typing as tp

import torch
from torch import nn

# from .. import quantization as qt

# Based on https://github.com/facebookresearch/audiocraft/blob/896ec7c47f5e5d1e5aa1e4b260c4405328bf009d/audiocraft/models/encodec.py#L28
class BaseAudioTokenizer(ABC, nn.Module):
    """Base API for all compression models that aim at being used as audio tokenizers
    with a language model.
    """

    #TODO: We are not using this for training yet
    # @abstractmethod
    # def forward(self, x: torch.Tensor) -> qt.QuantizedResult:
    #     ...

    @abstractmethod
    def encode(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        """See `EncodecModel.encode`."""
        ...

    @abstractmethod
    def decode(self, codes: torch.Tensor, scale: tp.Optional[torch.Tensor] = None):
        """See `EncodecModel.decode`."""
        ...

    @abstractmethod
    def decode_latent(self, codes: torch.Tensor):
        """Decode from the discrete codes to continuous latent space."""
        ...

    @property
    @abstractmethod
    def channels(self) -> int:
        ...

    @property
    @abstractmethod
    def frame_rate(self) -> float:
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        ...

    @property
    @abstractmethod
    def cardinality(self) -> int:
        ...

    @property
    @abstractmethod
    def num_codebooks(self) -> int:
        ...

    @property
    @abstractmethod
    def total_codebooks(self) -> int:
        ...

    @property
    def orig_vocab_size(self) -> int:
        ...

    @property
    def vocab_size(self) -> int:
        """ The original vocab_size + special tokens such as padding and EOS """
        return self.orig_vocab_size + 3

    @property
    def pad_token_id(self) -> int:
        return self.vocab_size - 1

    @property
    def bos_token_id(self) -> int:
        return self.vocab_size - 2

    @property
    def eos_token_id(self) -> int:
        return self.vocab_size - 3

    @abstractmethod
    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer."""
        ...

    @abstractmethod
    def get_pretrained(self, name: str, device: tp.Union[torch.device, str] = 'cpu') -> 'BaseAudioTokenizer':
        ...
