import torch
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Float
from typeguard import typechecked


@typechecked
def scaled_dot_product_attention(
    q: Float[torch.Tensor, "b l2 h d"],
    k: Float[torch.Tensor, "b l1 h d"],
    v: Float[torch.Tensor, "b l1 h d"],
    is_causal: bool = False,
) -> Float[torch.Tensor, "b l2 h d"]:
    d_model, len_q, len_k = q.size(-1), q.size(-3), k.size(-3)
    scale = 1.0 / (d_model**0.5)
    attn_weights = einsum(q, k, "b l1 h d, b l2 h d -> b h l1 l2") * scale

    attn_bias = torch.zeros(1, 1, len_q, len_k, dtype=q.dtype)
    if is_causal:
        attn_bias_msk = torch.ones(1, 1, len_q, len_k, dtype=torch.bool).tril(
            diagonal=0
        )
        attn_bias.masked_fill_(attn_bias_msk.logical_not(), float("-inf"))
        attn_bias.to(q.dtype)

    print(attn_weights.shape)
    print(attn_bias.shape)

    attn_weights += attn_bias
    attn_weights = attn_weights.softmax(dim=-1)
    # attn_weights = attn_weights.dropout(p=0.1)
    attn = einsum(attn_weights, v, "b h l1 l2, b l1 h d -> b l2 h d")
    print(attn.shape)
    return attn


@typechecked
def split_heads(
    x: Float[torch.Tensor, "b l d"], n_heads: int
) -> Float[torch.Tensor, "b l h d"]:
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
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        self.head_dim = d_model // n_heads
        assert (
            self.head_dim * n_heads == d_model
        ), f"{d_model =} must be divisible by {n_heads =}"

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)

    @typechecked
    def __call__(self, x: Float[torch.Tensor, "b l d"]) -> Float[torch.Tensor, "b l d"]:

        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q_h, k_h, v_h = map(lambda x: split_heads(x, self.n_heads), (q, k, v))

        out_h = scaled_dot_product_attention(q_h, k_h, v_h, is_causal=False)
        return merge_heads(out_h, self.n_heads)

        # out = attention(q, k, v, mask=mask)
