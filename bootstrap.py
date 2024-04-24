import random

import torch
import transformers as hgf

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