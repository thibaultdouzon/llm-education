from enum import Enum

import torch
from einops import einsum, rearrange
from jaxtyping import Float, Int
from loguru import logger
from typeguard import typechecked


class GenerationStrategies(Enum):
    DETERMINIST = 0
    SAMPLING = 1
    BEAM_SEARCH = 2


eps = 1e-6


def softmax_temp(
    x: Float[torch.Tensor, "b l d"],
    dim: int = -1,
    temperature: float = 0,
) -> Float[torch.Tensor, "b l d"]:
    """
    Compute the softmax with temperature applied.
    Input values are normalized to avoid overflows due to exp.
    """
    if temperature < eps:
        return x.argmax(dim=dim)
    maximum = torch.max(x / temperature)
    exps_norm = torch.exp(x / temperature - maximum)
    return exps_norm / exps_norm.sum(dim=dim)


@typechecked
def generate_beam_search(
    model,
    x: Int[torch.Tensor, "b l"],
    n_tokens: int = 100,
    n_beams: int = 3,
    temperature: float = 0.0,
    *,
    return_log_scores: bool = False,
) -> (
    Int[torch.Tensor, "b ll"]
    | tuple[Int[torch.Tensor, "b ll"], Float[torch.Tensor, "b"]]
):
    """
    Beam search algorithm.
    Implementation notes: generated sequences might diverge, must keep track of all of them
    assert batch_size is 1 for now.
    """
    scores = torch.ones(1)
    for i in range(n_tokens):
        logger.debug(f"Step {i}")
        logits = model(x)[:, -1].softmax(dim=-1)  # TODO use log softmax and sums
        topk_gen = torch.topk(logits, n_beams, dim=-1)
        new_scores = rearrange(
            einsum(scores, topk_gen.values, "b, b k -> b k"), "b k -> (b k)"
        )
        topk_scores = torch.topk(new_scores, n_beams, dim=-1)

        topk_scores_indices = topk_scores.indices[:n_beams]

        selected_tokens = rearrange(topk_gen.indices, "b k -> (b k)")[
            topk_scores_indices
        ]
        selected_sequences = topk_scores_indices // n_beams
        selected_tokens_scores = new_scores[topk_scores_indices]

        logger.info(f"{topk_gen=}")
        logger.info(f"{new_scores=}")
        logger.info(f"{topk_scores=}")
        logger.info(f"{topk_scores_indices=}")
        logger.info(f"{selected_tokens=}")
        logger.info(f"{selected_sequences=}")
        logger.info(f"{selected_tokens_scores=}")

        scores = selected_tokens_scores
        if x.size(0) == 1:
            x = x.repeat(n_beams, 1)

        x = torch.cat(
            [
                x[selected_sequences],
                selected_tokens.unsqueeze(-1),
            ],
            dim=1,
        )
        logger.info(f"{x=}")

    return torch.randint(0, model.config.d_vocab, (x.size(0), 3))


@typechecked
def generate_greedy(
    self,
    x: Int[torch.Tensor, "b l"],
    n_tokens: int = 100,
    temperature: float = 0.0,
    *,
    return_log_scores: bool = False,
) -> (
    Int[torch.Tensor, "b ll"]
    | tuple[Int[torch.Tensor, "b ll"], Float[torch.Tensor, "b"]]
):
    for i in range(n_tokens):
        logits = self(x)[:, -1]
        if temperature < eps:
            pred = logits.argmax(dim=-1)
        else:
            sampling_weights = softmax_temp(logits, dim=-1, temperature=temperature)
            pred = torch.multinomial(sampling_weights, num_samples=1)

        x = torch.cat([x, pred.unsqueeze(-1)], dim=1)
        # TODO: EOS token ?

    return x
