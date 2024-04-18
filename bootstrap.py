import torch
import transformers as hgf

from src.models import transformer
from src.utils import sampling

model = transformer.Transformer.from_pretrained("gpt2").eval()
tok = hgf.GPT2Tokenizer.from_pretrained("gpt2")


def generate_rnd(model, **kwargs):
    size = kwargs.get("size", (1, 1))
    n_tokens = kwargs.get("n_tokens", 2)
    n_beams = kwargs.get("n_beams", 3)
    strategy = kwargs.get("strategy", sampling.GenerationStrategies.BEAM_SEARCH)
    return model.generate(
        torch.randint(0, model.config.d_vocab, size),
        n_tokens=n_tokens,
        n_beams=n_beams,
        strategy=strategy,
    )
