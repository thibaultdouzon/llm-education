import random

import torch
import transformers as hgf
from jaxtyping import Int

from src.models import transformer
from src.utils import sampling

model = transformer.Transformer.from_pretrained("gpt2").eval()
tok = hgf.GPT2Tokenizer.from_pretrained("gpt2")


def generate_rnd(model, **kwargs) -> Int[torch.Tensor, "b l"]:
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


def generate_prompt(model, tok, prompt: str, **kwargs) -> str:
    if "seed" in kwargs:
        torch.manual_seed(kwargs["seed"])

    in_tokens = tok.encode(prompt, return_tensors="pt")

    n_tokens = kwargs.get("n_tokens", 10)
    n_beams = kwargs.get("n_beams", 1)
    temperature = kwargs.get("temperature", 0.0)
    strategy = kwargs.get("strategy", sampling.GenerationStrategies.DETERMINIST)
    return_log_scores = kwargs.get("return_log_scores", False)

    print(in_tokens.shape)

    out_tokens = model.generate(
        in_tokens,
        n_tokens=n_tokens,
        n_beams=n_beams,
        temperature=temperature,
        strategy=strategy,
        return_log_scores=return_log_scores,
    )

    if return_log_scores:
        return tok.decode(
            out_tokens.tokens[0], skip_special_tokens=True
        ), out_tokens.log_prob
    return tok.decode(out_tokens[0], skip_special_tokens=True)
