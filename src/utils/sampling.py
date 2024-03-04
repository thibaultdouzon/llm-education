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
    """
    Compute the softmax with temperature applied.
    Input values are normalized to avoid overflows due to exp.
    """
    if temperature < eps:
        return input.argmax(dim=dim)
    maximum = torch.max(input / temperature)
    exps_norm = torch.exp(input / temperature - maximum)
    return exps_norm / exps_norm.sum(dim=dim)
