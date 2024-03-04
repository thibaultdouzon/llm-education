from enum import Enum

import torch
from jaxtyping import Float


class GenerationStrategies(Enum):
    DETERMINIST = 0
    SAMPLING = 1
    BEAM_SEARCH = 2


eps = 1e-6


def softmax_temp(
    input: Float[torch.Tensor, "b l d"], dim: int = -1, temperature: float = 0
) -> Float[torch.Tensor, "b l d"]:
    if temperature < eps:
        return input.argmax(dim=dim)
    exps = torch.exp(input / temperature)
    return exps / exps.sum(dim=dim)
