import torch
import transformers

from src.models import transformer

t_hf = transformers.GPT2LMHeadModel.from_pretrained("gpt2").eval()
t = transformer.Transformer.from_pretrained("gpt2").eval()


def get_input(
    batch: int = 2,
    sequence_len: int = 10,
    vocab_size: int = 50257,
):
    return torch.randint(0, vocab_size, (batch, sequence_len))
