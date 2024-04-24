import random

import torch
import transformers as hgf
from jaxtyping import Int

from src.models import transformer
from src.utils import sampling

model = transformer.Transformer.from_pretrained("gpt2").eval()
tok = hgf.GPT2Tokenizer.from_pretrained("gpt2")


def generate_rnd(model, **kwargs):
    if "seed" in kwargs:
        torch.manual_seed(kwargs["seed"])

    size = kwargs.get("size", (1, 1))
    prompt = torch.tensor(
        kwargs.get("prompt", random.randint(0, model.config.d_vocab))
    ).view(size)
    n_tokens = kwargs.get("n_tokens", 2)
    n_beams = kwargs.get("n_beams", 3)
    temperature = kwargs.get("temperature", 0.0)
    strategy = kwargs.get("strategy", sampling.GenerationStrategies.BEAM_SEARCH)
    return_log_scores = kwargs.get("return_log_scores", False)
    return model.generate(
        prompt,
        n_tokens=n_tokens,
        n_beams=n_beams,
        temperature=temperature,
        strategy=strategy,
        return_log_scores=return_log_scores,
    )


def score_generation(model, sequence: Int[torch.Tensor, "b l"], **kwargs):
    if "seed" in kwargs:
        torch.manual_seed(kwargs["seed"])

    temperature = kwargs.get("temperature", 0.0)

    assert sequence.size(0) == 1
    seq_len = sequence.size(1)

    with torch.inference_mode():
        logits = model(sequence)  # b l d

        if temperature < sampling.eps:
            logits_log_scores = torch.log_softmax(logits, dim=-1)
        else:
            logits_log_scores = sampling.log_softmax_temp(
                logits, temperature=temperature
            )

    log_scores = 0.0

    for i in range(seq_len - 1):
        log_scores += logits_log_scores[0, i, sequence[0, i + 1]]

    return log_scores
