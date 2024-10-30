from dataclasses import dataclass
from functools import partial

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import torch
from beartype import beartype
from einops import einsum, rearrange
from jaxtyping import Array, Float, Int, PRNGKeyArray, jaxtyped
from transformers import AutoConfig, AutoModelForCausalLM, activations

from src.utils.sampling import GenerationStrategies, generate_beam_search, generate_greedy, log_softmax_temp


@dataclass(frozen=True)
class Config:
    d_model: int
    d_vocab: int
    max_size: int
    d_ff: int
    n_heads: int
    n_layers: int
    is_causal: bool
    dropout: float

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, f"{self.d_model =} must be divisible by {self.n_heads = }"


class DotProductAttention(eqx.Module):
    config: Config
    attn_dropout: nn.Dropout

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.attn_dropout = nn.Dropout(config.dropout)

    @eqx.filter_jit
    @jaxtyped(typechecker=beartype)  # type: ignore
    def __call__(
        self,
        q: Float[Array, "length_q d_model_split"],
        k: Float[Array, "lenght_kv d_model_split"],
        v: Float[Array, "length_kv d_model_split"],
        key: PRNGKeyArray | None = None,
        inference_mode: bool = False,
    ) -> Float[Array, "length_q d_model_split"]:
        d_model, len_q, len_k = q.shape[1], q.shape[0], k.shape[0]
        scale = 1.0 / (d_model**0.5)
        attn_weights = einsum(q, k, "lq e, lkv e -> lq lkv") * scale

        attn_bias = jnp.zeros((len_q, len_k), dtype=q.dtype)
        if self.config.is_causal:
            attn_bias_msk = jnp.tril(jnp.ones((len_q, len_k), dtype=jnp.bool))
            attn_bias = attn_bias + jnp.where(attn_bias_msk, 0.0, -1e9)

        attn_weights = attn_weights + attn_bias

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_weights = self.attn_dropout(attn_weights, key=key, inference=inference_mode)
        attn: Float[Array, "length_kv d_model_split"] = einsum(attn_weights, v, "lq lkv, lkv e -> lq e")
        return attn


# class SinCosPositionalEncoding(eqx.Module):
#     config: Config
#     pe: Array

#     def __init__(self, config: Config):
#         self.config = config

#         self.pe = jnp.zeros(
#             (config.max_size, config.d_model),
#             dtype=jnp.float32,
#         )
#         self.init_weights()
#         print(self.pe.shape)

#     def init_weights(self) -> None:
#         pos = jnp.arange(0, self.config.max_size, dtype=self.pe.dtype)[:, None]
#         pos = pos * jnp.exp(jnp.arange(0, self.config.d_model, 2) * (-jnp.log(10000.0) / self.config.d_model))
#         self.pe = rearrange(
#             [jnp.sin(pos), jnp.cos(pos)],
#             "d2 l d -> l (d d2)",
#             d=self.config.d_model // 2,
#             d2=2,
#             l=self.config.max_size,
#         )

#     @eqx.filter_jit
#     @jaxtyped(typechecker=beartype)
#     def __call__(self, x: Float[Array, "length d_model"]) -> Float[Array, "length d_model"]:
#         x = x + self.pe[: x.shape[1], :]
#         return x


@eqx.filter_jit
@jaxtyped(typechecker=beartype)  # type: ignore
def split_heads(x: Float[Array, "length d_model"], n_heads: int) -> Float[Array, "n_heads length d_model_split"]:
    l, d = x.shape
    assert d % n_heads == 0, f"{d = } must be divisible by {n_heads = }"
    head_dim = d // n_heads
    return rearrange(x, "l (h e) -> h l e", h=n_heads, e=head_dim)


@eqx.filter_jit
@jaxtyped(typechecker=beartype)  # type: ignore
def merge_heads(x: Float[Array, "n_heads length d_model_split"], n_heads: int) -> Float[Array, "length d_model"]:
    h, l, d = x.shape
    assert h == n_heads, f"{h =} must be equal to {n_heads =}"
    return rearrange(x, "h l e -> l (h e)")


class SelfAttention(eqx.Module):
    config: Config
    qkv_proj: nn.Linear
    attention: DotProductAttention
    out_proj: nn.Linear
    out_dropout: nn.Dropout

    def __init__(self, config: Config, key: PRNGKeyArray):
        super().__init__()

        self.config = config

        head_dim = config.d_model // config.n_heads
        assert (
            head_dim * config.n_heads == config.d_model
        ), f"{config.d_model =} must be divisible by {config.n_heads =}"

        key, key_qkv, key_out = jax.random.split(key, 3)

        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, key=key_qkv)

        self.attention = DotProductAttention(config)

        self.out_proj = nn.Linear(config.d_model, config.d_model, key=key_out)
        self.out_dropout = nn.Dropout(config.dropout)

    @eqx.filter_jit
    @jaxtyped(typechecker=beartype)  # type: ignore
    def __call__(
        self,
        x: Float[Array, "length d_model"],
        key: PRNGKeyArray | None = None,
        inference_mode: bool = False,
    ) -> Float[Array, "length d_model"]:
        key_attn, key_drop = None, None
        if not inference_mode:
            assert key is not None, f"{key = } must be provided during training"
            key_attn, key_drop = jax.random.split(key)

        x = jax.vmap(self.qkv_proj)(x)
        q, k, v = jnp.split(x, 3, axis=-1)
        q_h, k_h, v_h = map(
            partial(split_heads, n_heads=self.config.n_heads),
            (q, k, v),
        )

        out_h = jax.vmap(partial(self.attention, key=key_attn, inference_mode=inference_mode), in_axes=0)(q_h, k_h, v_h)
        out = merge_heads(out_h, self.config.n_heads)

        out = jax.vmap(self.out_proj)(out)
        out = self.out_dropout(out, key=key_drop, inference=inference_mode)
        return out


# class FeedForward(eqx.Module):
#     def __init__(self, config: Config):
#         super().__init__()
#         self.config = config

#         self.m_ff_proj = nn.Linear(config.d_model, config.d_ff)
#         self.act = activations.NewGELUActivation()
#         self.ff_m_proj = nn.Linear(config.d_ff, config.d_model)
#         self.ff_dropout = nn.Dropout(config.dropout)

#     @jaxtyped(typechecker=beartype)
#     def forward(self, x: Float[Array, "b l d"]) -> Float[Array, "b l d"]:
#         x = self.m_ff_proj(x)
#         x = self.act(x)
#         x = self.ff_m_proj(x)
#         return self.ff_dropout(x)


# class Block(eqx.Module):
#     def __init__(self, config: Config):
#         super().__init__()
#         self.config = config
#         self.attn = SelfAttention(config)
#         self.norm_attn = nn.LayerNorm(config.d_model)
#         self.ffn = FeedForward(config)
#         self.norm_ffn = nn.LayerNorm(config.d_model)

#     @jaxtyped(typechecker=beartype)
#     def forward(self, x: Float[Array, "b l d"]) -> Float[Array, "b l d"]:
#         x = self.attn(self.norm_attn(x)) + x

#         x = self.ffn(self.norm_ffn(x)) + x
#         return x


# class Transformer(eqx.Module):
#     def __init__(
#         self,
#         config: Config,
#     ):
#         super().__init__()
#         self.config = config

#         self.embedding = nn.Embedding(config.d_vocab, config.d_model)
#         self.positional_encoding = nn.Embedding(config.max_size, config.d_model)

#         self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
#         self.norm_transformer = nn.LayerNorm(config.d_model)
#         self.lm_head = nn.Linear(config.d_model, config.d_vocab, bias=False)

#         self.apply(self._init_weights)

#     def _init_weights(self, module: eqx.Module):
#         if isinstance(module, nn.Linear):
#             torch.nn.init.xavier_normal_(module.weight)
#             if module.bias is not None:
#                 torch.nn.init.zeros_(module.bias)
#         elif isinstance(module, nn.Embedding):
#             torch.nn.init.xavier_normal_(module.weight)
#         elif isinstance(module, (nn.LayerNorm,)):
#             torch.nn.init.zeros_(module.bias)
#             torch.nn.init.ones_(module.weight)

#     @jaxtyped(typechecker=beartype)
#     def forward(self, x: Int[Array, "b l"]) -> Float[Array, "b l d"]:
#         x = self.embedding(x) + self.positional_encoding(
#             torch.arange(x.size(1), device=x.device, dtype=torch.int64).unsqueeze(0)
#         )
#         for layer in self.layers:
#             x = layer(x)

#         x = self.norm_transformer(x)
#         return self.lm_head(x)

#     @jaxtyped(typechecker=beartype)
#     def score_sequences(
#         self,
#         x: Int[Array, "batch l"],
#         temperature: float = 0.0,
#     ) -> Float[Array, "batch"]:
#         # TODO: Correctly deal with EOS token
#         seq_len = x.size(1)

#         with torch.inference_mode():
#             logits = self(x)  # b l d

#             logits_log_scores = log_softmax_temp(logits, dim=-1, temperature=temperature)

#         log_scores = torch.zeros(x.size(0))
#         batches_indices = torch.arange(0, x.size(0))
#         for i in range(seq_len - 1):
#             log_scores += logits_log_scores[batches_indices, i, x[batches_indices, i + 1]]

#         return log_scores

#     @jaxtyped(typechecker=beartype)
#     def generate(
#         self,
#         x: Int[Array, "batch l"],
#         n_tokens: int = 100,
#         n_beams: int = 1,
#         strategy: GenerationStrategies = GenerationStrategies.DETERMINIST,
#         temperature: float = 0.0,
#         *,
#         return_log_scores: bool = False,
#     ) -> Int[Array, "batch ll"] | tuple[Int[Array, "batch ll"], Float[Array, "batch"]]:
#         with torch.inference_mode():
#             match strategy:
#                 case GenerationStrategies.DETERMINIST:
#                     assert (
#                         temperature == 0.0
#                     ), f"{strategy = } and {temperature = } are incompatible, temperature must be 0.0"
#                     return generate_greedy(
#                         self,
#                         x,
#                         n_tokens,
#                         temperature=0.0,
#                         return_log_scores=return_log_scores,
#                     )
#                 case GenerationStrategies.SAMPLING:
#                     return generate_greedy(
#                         self,
#                         x,
#                         n_tokens,
#                         temperature=temperature,
#                         return_log_scores=return_log_scores,
#                     )
#                 case GenerationStrategies.BEAM_SEARCH:
#                     return generate_beam_search(
#                         self,
#                         x,
#                         n_tokens,
#                         n_beams,
#                         temperature=temperature,
#                         return_log_scores=return_log_scores,
#                     )
#                 case _:
#                     raise NotImplementedError()

#     @classmethod
#     def from_pretrained(
#         cls,
#         model_name: str,
#     ):
#         config_hf = AutoConfig.from_pretrained(model_name)

#         config = Config(
#             d_model=config_hf.hidden_size,
#             d_vocab=config_hf.vocab_size,
#             max_size=config_hf.n_positions,
#             d_ff=4 * config_hf.hidden_size,
#             n_heads=config_hf.n_head,
#             n_layers=config_hf.n_layer,
#             is_causal=True,
#             dropout=config_hf.attn_pdrop,
#         )

#         gpt2_layer_translation = {
#             "transformer.": "",
#             "wte": "embedding",
#             "wpe": "positional_encoding",
#             "h.": "layers.",
#             "ln_1": "norm_attn",
#             "ln_2": "norm_ffn",
#             "attn.c_attn": "attn.qkv_proj",
#             "attn.c_proj": "attn.out_proj",
#             "mlp.c_fc": "ffn.m_ff_proj",
#             "mlp.c_proj": "ffn.ff_m_proj",
#             "ln_f": "norm_transformer",
#         }

#         model_hf = AutoModelForCausalLM.from_pretrained(model_name)
#         sd_hf = model_hf.state_dict()

#         model = cls(config)
#         sd = model.state_dict()

#         transposed = [
#             "attn.c_attn.weight",
#             "attn.c_proj.weight",
#             "mlp.c_fc.weight",
#             "mlp.c_proj.weight",
#         ]
#         copied_layers = set()
#         for k_hf, v_hf in sd_hf.items():
#             k = k_hf
#             for tr_hf, tr in gpt2_layer_translation.items():
#                 if tr_hf in k:
#                     k = k.replace(tr_hf, tr)
#             if any(t in k_hf for t in transposed):
#                 v_hf = v_hf.transpose(0, 1)
#             assert k in sd, f"{k = } not in {sd.keys()}"
#             assert sd[k].shape == v_hf.shape, f"{sd[k].shape = } != {v_hf.shape = }"

#             with torch.no_grad():
#                 sd[k].copy_(v_hf)
#             copied_layers.add(k)

#         return model.eval()
