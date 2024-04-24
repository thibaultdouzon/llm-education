from collections import namedtuple
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

SamplingResult = namedtuple("SamplingResult", ["tokens", "log_prob"])


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


def log_softmax_temp(
    x: Float[torch.Tensor, "b l d"],
    dim: int = -1,
    temperature: float = 0,
) -> Float[torch.Tensor, "b l d"]:
    """
    Compute the log softmax with temperature applied.
    Use the logsumexp trick to avoid overflows (https://stackoverflow.com/questions/61567597/how-is-log-softmax-implemented-to-compute-its-value-and-gradient-with-better).
    """
    if temperature < eps:
        return x.log_softmax(dim=dim)
    maximum = torch.max(x / temperature)
    log_sum_exp = torch.logsumexp(x / temperature - maximum, dim=dim)
    return x / temperature - maximum - log_sum_exp


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
    for _ in range(n_tokens):
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

    best_score_idx = scores.argmax()
    best_score = scores[best_score_idx]
    best_sequence = x[best_score_idx, :].unsqueeze(0)

    if return_log_scores:
        return SamplingResult(tokens=best_sequence, log_prob=best_score.log())
    return best_sequence


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
    if x.size(0) > 1:
        raise NotImplementedError("Batch size > 1 not implemented yet")
    log_prob = 0.0
    for i in range(n_tokens):
        logits = self(x)[:, -1]
        if temperature < eps:
            log_prob_logits = logits.log_softmax(dim=-1)
            pred = torch.argmax(log_prob_logits)
            log_prob += log_prob_logits[:, pred]
        else:
            log_prob_logits = log_softmax_temp(logits, dim=-1, temperature=temperature)
            pred = torch.multinomial(log_prob_logits.exp(), num_samples=1).squeeze()
            log_prob += log_prob_logits[:, pred]

        x = torch.cat([x, pred.view((1, 1))], dim=1)
        # TODO: EOS token ?

    if return_log_scores:
        return SamplingResult(tokens=x, log_prob=log_prob)
    return x
