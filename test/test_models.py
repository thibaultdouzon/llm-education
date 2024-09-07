import jax
import jax.numpy as jnp
import pytest
import torch
import transformers as hgf
from jaxtyping import Float, Int

from src.models import transformer

# @pytest.fixture
# def gpt2_model() -> transformer.Transformer:
#     model = transformer.Transformer.from_pretrained("gpt2").eval()
#     return model


# @pytest.fixture
# def hgf_gpt2_model() -> hgf.GPT2LMHeadModel:
#     model = hgf.GPT2LMHeadModel.from_pretrained("gpt2").eval()
#     return model


def dummy_input(batch: int, sequence_len: int, vocab_size: int) -> Int[torch.Tensor, "b l"]:
    return torch.randint(0, vocab_size, (batch, sequence_len), dtype=torch.long)


class TestSinCosPositionalEncoding:
    def test_shape_out(self):
        config = transformer.Config(
            d_model=768, d_vocab=50257, max_size=1024, d_ff=3072, n_heads=12, n_layers=12, is_causal=True, dropout=0.1
        )
        pe = transformer.SinCosPositionalEncoding(config)

        key = jax.random.PRNGKey(0)
        dummy_in = jax.random.normal(key, shape=(2, 64, config.d_model), dtype=jnp.float32)
        pe_out = pe(dummy_in)
        assert pe_out.shape == (2, 64, config.d_model)


class TestGPT2:
    def test_shape_out(self, gpt2_model):
        vocab_size = gpt2_model.config.d_vocab
        x = dummy_input(2, 10, vocab_size)
        y = gpt2_model(x)
        assert y.shape == (2, 10, vocab_size)

    def test_identic_output_hgf(self, gpt2_model, hgf_gpt2_model):
        x = dummy_input(2, 10, gpt2_model.config.d_vocab)
        y = gpt2_model(x)
        y_hgf = hgf_gpt2_model(x).logits
        assert torch.allclose(y, y_hgf, atol=1e-4)
