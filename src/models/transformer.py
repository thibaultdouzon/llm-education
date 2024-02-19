from functools import partial

import torch
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Float, Int
from typeguard import typechecked


class SinCosPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model), requires_grad=False)

    def init_weights(self):
        pos = torch.arange(self.max_len, dtype=self.pe.dtype)
        pos = pos.unsqueeze(-1) / (
            10000 ** (torch.arange(0, self.d_model, 2) / self.d_model)
        )
        print(torch.sin(pos[:10]))
        self.pe.data = rearrange(
            [torch.sin(pos), torch.cos(pos)],
            "a l d2 -> 1 l (d2 a)",
            a=2,
            l=self.max_len,
            d2=self.d_model // 2,
        )

    @typechecked
    def forward(self, x: Float[torch.Tensor, "b l d"]) -> Float[torch.Tensor, "b l d"]:
        x = x + self.pe[:, : x.size(1)]
        return x


class DotProductAttention(nn.Module):
    def __init__(self, is_causal: bool = False, dropout: float = 0.1):
        super().__init__()
        self.dropout = dropout
        self.is_causal = is_causal

        self.attn_dropout = nn.Dropout(dropout)

    @typechecked
    def forward(
        self,
        q: Float[torch.Tensor, "b l2 h e"],
        k: Float[torch.Tensor, "b l1 h e"],
        v: Float[torch.Tensor, "b l1 h e"],
    ) -> Float[torch.Tensor, "b l2 h e"]:
        d_model, len_q, len_k = q.size(-1), q.size(-3), k.size(-3)
        scale = 1.0 / (d_model**0.5)
        attn_weights = einsum(q, k, "b l2 h e, b l1 h e -> b h l2 l1") * scale

        attn_bias = torch.zeros(1, 1, len_q, len_k, dtype=q.dtype)
        if self.is_causal:
            attn_bias_msk = torch.ones(1, 1, len_q, len_k, dtype=torch.bool).tril(
                diagonal=0
            )
            attn_bias.masked_fill_(attn_bias_msk.logical_not(), float("-inf"))
            attn_bias.to(q.dtype)

        attn_weights += attn_bias
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn = einsum(attn_weights, v, "b h l2 l1, b l1 h e -> b l2 h e")
        return attn


@typechecked
def split_heads(
    x: Float[torch.Tensor, "b l d"], n_heads: int
) -> Float[torch.Tensor, "b l h e"]:
    b, l, d = x.size()
    assert d % n_heads == 0, f"{d =} must be divisible by {n_heads =}"
    head_dim = d // n_heads
    return rearrange(x, "b l (h e) -> b l h e", h=n_heads, e=head_dim)


@typechecked
def merge_heads(
    x: Float[torch.Tensor, "b l h e"], n_heads: int
) -> Float[torch.Tensor, "b l d"]:
    b, l, h, d = x.size()
    assert h == n_heads, f"{h =} must be equal to {n_heads =}"
    return rearrange(x, "b l h e -> b l (h e)")


class SelfAttention(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, is_causal: bool = False, dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.is_causal = is_causal
        self.dropout = dropout

        self.head_dim = d_model // n_heads
        assert (
            self.head_dim * n_heads == d_model
        ), f"{d_model =} must be divisible by {n_heads =}"

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)

        self.attention = DotProductAttention(is_causal=False, dropout=dropout)

        self.out_proj = nn.Linear(d_model, d_model)
        self.out_dropout = nn.Dropout(dropout)

    @typechecked
    def forward(self, x: Float[torch.Tensor, "b l d"]) -> Float[torch.Tensor, "b l d"]:

        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q_h, k_h, v_h = map(partial(split_heads, n_heads=self.n_heads), (q, k, v))

        out_h = self.attention(q_h, k_h, v_h)
        out = merge_heads(out_h, self.n_heads)
        out = self.out_proj(out)
        return self.out_dropout(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout

        self.m_ff_proj = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.ff_m_proj = nn.Linear(d_ff, d_model)
        self.ff_dropout = nn.Dropout(dropout)

    @typechecked
    def forward(self, x: Float[torch.Tensor, "b l d"]) -> Float[torch.Tensor, "b l d"]:
        x = self.m_ff_proj(x)
        x = self.act(x)
        x = self.ff_m_proj(x)
        return self.ff_dropout(x)


class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int,
        is_causal: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.is_causal = is_causal
        self.dropout = dropout

        self.attn = SelfAttention(d_model, n_heads, is_causal, dropout)
        self.norm_attn = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm_ffn = nn.LayerNorm(d_model)

    @typechecked
    def forward(self, x: Float[torch.Tensor, "b l d"]) -> Float[torch.Tensor, "b l d"]:
        x += self.attn(self.norm_attn(x))
        return x + self.ffn(self.norm_ffn(x))


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_vocab: int,
        d_ff: int,
        n_heads: int,
        n_layers: int,
        is_causal: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(d_vocab, d_model)
        self.positional_encoding = nn.Embedding(d_vocab, d_model)

        self.layers = nn.ModuleList(
            [
                Block(d_model, d_ff, n_heads, is_causal=is_causal, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

    @typechecked
    def forward(self, x: Int[torch.Tensor, "b l"]) -> Float[torch.Tensor, "b l d"]:
        x = self.embedding(x) + self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return x
