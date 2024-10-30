from functools import partial

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
    def test_shape_out(self) -> None:
        config = transformer.Config(
            d_model=768, d_vocab=50257, max_size=1024, d_ff=3072, n_heads=12, n_layers=12, is_causal=True, dropout=0.1
        )
        pe = transformer.SinCosPositionalEncoding(config)

        key = jax.random.PRNGKey(0)
        layer_in = jax.random.normal(key, shape=(2, 64, config.d_model), dtype=jnp.float32)
        layer_out = pe(layer_in)
        assert layer_out.shape == (2, 64, config.d_model)


class TestDotProductAttention:
    def test_shape_out(self) -> None:
        config = transformer.Config(
            d_model=768, d_vocab=50257, max_size=1024, d_ff=3072, n_heads=12, n_layers=12, is_causal=True, dropout=0.1
        )
        attention = transformer.DotProductAttention(config)

        key = jax.random.PRNGKey(0)
        key, k_q, k_k, k_v = jax.random.split(key, 4)
        layer_in_q = jax.random.normal(k_q, shape=(64, config.d_model), dtype=jnp.float32)
        layer_in_k = jax.random.normal(k_k, shape=(64, config.d_model), dtype=jnp.float32)
        layer_in_v = jax.random.normal(k_v, shape=(64, config.d_model), dtype=jnp.float32)

        key, k_call = jax.random.split(key)

        layer_out = attention(layer_in_q, layer_in_k, layer_in_v, key=k_call)

        assert layer_out.shape == (64, config.d_model)

    def test_similar_vectors(self) -> None:
        config = transformer.Config(
            d_model=12, d_vocab=50257, max_size=1024, d_ff=3072, n_heads=12, n_layers=12, is_causal=True, dropout=0.1
        )
        attention = transformer.DotProductAttention(config)

        key = jax.random.PRNGKey(0)
        key, k_q, k_k, k_v = jax.random.split(key, 4)

        layer_in_q = jax.random.normal(k_q, shape=(2, config.d_model), dtype=jnp.float32)
        layer_in_k = jnp.tile(jax.random.normal(k_k, shape=(1, config.d_model), dtype=jnp.float32), (2, 1))

        layer_in_v = jax.random.normal(k_v, shape=(2, config.d_model), dtype=jnp.float32)

        assert layer_in_k.shape == layer_in_q.shape == layer_in_v.shape == (2, config.d_model)

        key, k_call = jax.random.split(key)

        layer_out = attention(layer_in_q, layer_in_k, layer_in_v, inference_mode=True)

        expected_out = jnp.array([layer_in_v[0, :], jnp.sum(layer_in_v, axis=0) / 2])

        assert layer_out.shape == (2, config.d_model)
        assert jnp.allclose(layer_out, expected_out, atol=1e-9)


class TestSplitHeads:
    def test_shape_out(self) -> None:
        config = transformer.Config(
            d_model=768, d_vocab=50257, max_size=1024, d_ff=3072, n_heads=12, n_layers=12, is_causal=True, dropout=0.1
        )
        key = jax.random.PRNGKey(0)
        layer_in = jax.random.normal(key, shape=(64, config.d_model), dtype=jnp.float32)
        layer_out = transformer.split_heads(layer_in, n_heads=config.n_heads)
        assert layer_out.shape == (config.n_heads, 64, config.d_model / config.n_heads)


class TestMergeHeads:
    def test_shape_out(self) -> None:
        config = transformer.Config(
            d_model=768, d_vocab=50257, max_size=1024, d_ff=3072, n_heads=12, n_layers=12, is_causal=True, dropout=0.1
        )
        key = jax.random.PRNGKey(0)
        layer_in = jax.random.normal(
            key, shape=(config.n_heads, 64, config.d_model // config.n_heads), dtype=jnp.float32
        )
        layer_out = transformer.merge_heads(layer_in, n_heads=config.n_heads)
        assert layer_out.shape == (64, config.d_model)

    def test_all_round_split_merge(self) -> None:
        config = transformer.Config(
            d_model=768, d_vocab=50257, max_size=1024, d_ff=3072, n_heads=12, n_layers=12, is_causal=True, dropout=0.1
        )
        key = jax.random.PRNGKey(0)
        layer_in = jax.random.normal(key, shape=(64, config.d_model), dtype=jnp.float32)
        layer_out = transformer.merge_heads(
            transformer.split_heads(layer_in, n_heads=config.n_heads), n_heads=config.n_heads
        )
        assert jnp.allclose(layer_in, layer_out, atol=1e-9)


class TestSelfAttention:
    def test_shape_out(self) -> None:
        config = transformer.Config(
            d_model=768, d_vocab=50257, max_size=1024, d_ff=3072, n_heads=12, n_layers=12, is_causal=True, dropout=0.1
        )
        key = jax.random.PRNGKey(0)
        k_attn, k_in, k_call = jax.random.split(key, 3)

        self_attention = transformer.SelfAttention(config, key=k_attn)

        layer_in = jax.random.normal(k_in, (64, config.d_model))
        layer_out = self_attention(layer_in, inference_mode=True)

        assert layer_out.shape == (64, config.d_model)


class TestGPT2:
    def test_shape_out(self, gpt2_model) -> None:
        vocab_size = gpt2_model.config.d_vocab
        x = dummy_input(2, 10, vocab_size)
        y = gpt2_model(x)
        assert y.shape == (2, 10, vocab_size)

    def test_identic_output_hgf(self, gpt2_model, hgf_gpt2_model) -> None:
        x = dummy_input(2, 10, gpt2_model.config.d_vocab)
        y = gpt2_model(x)
        y_hgf = hgf_gpt2_model(x).logits
        assert torch.allclose(y, y_hgf, atol=1e-4)
