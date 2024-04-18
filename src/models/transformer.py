from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from jaxtyping import Float, Int
from transformers import AutoConfig, AutoModelForCausalLM, activations
from typeguard import typechecked

from src.utils.sampling import (
    GenerationStrategies,
    generate_beam_search,
    generate_greedy,
)


@dataclass
class Config:
    d_model: int
    d_vocab: int
    max_size: int
    d_ff: int
    n_heads: int
    n_layers: int
    is_causal: bool
    dropout: float

    def __post_init__(self):
        assert (
            self.d_model % self.n_heads == 0
        ), f"{self.d_model =} must be divisible by {self.n_heads = }"


class SinCosPositionalEncoding(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.pe = nn.Parameter(
            torch.zeros(1, config.max_size, config.d_model),
            requires_grad=False,
        )

    def init_weights(self):
        pos = torch.arange(self.config.max_size, dtype=self.pe.dtype)
        pos = pos.unsqueeze(-1) / (
            10000 ** (torch.arange(0, self.config.d_model, 2) / self.config.d_model)
        )
        self.pe.data = rearrange(
            [torch.sin(pos), torch.cos(pos)],
            "a l d2 -> 1 l (d2 a)",
            a=2,
            l=self.config.max_size,
            d2=self.config.d_model // 2,
        )

    @typechecked
    def forward(self, x: Float[torch.Tensor, "b l d"]) -> Float[torch.Tensor, "b l d"]:
        x = x + self.pe[:, : x.size(1)]
        return x


class DotProductAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.attn_dropout = nn.Dropout(config.dropout)

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

        attn_bias = torch.zeros(1, 1, len_q, len_k, dtype=q.dtype, device=q.device)
        if self.config.is_causal:
            attn_bias_msk = torch.ones(
                1, 1, len_q, len_k, dtype=torch.bool, device=q.device
            ).tril(diagonal=0)
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
    assert d % n_heads == 0, f"{d = } must be divisible by {n_heads = }"
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
    def __init__(self, config: Config):
        super().__init__()

        self.config = config

        head_dim = config.d_model // config.n_heads
        assert (
            head_dim * config.n_heads == config.d_model
        ), f"{config.d_model =} must be divisible by {config.n_heads =}"

        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)

        self.attention = DotProductAttention(config)

        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.out_dropout = nn.Dropout(config.dropout)

    @typechecked
    def forward(self, x: Float[torch.Tensor, "b l d"]) -> Float[torch.Tensor, "b l d"]:
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q_h, k_h, v_h = map(
            partial(split_heads, n_heads=self.config.n_heads),
            (q, k, v),
        )

        out_h = self.attention(q_h, k_h, v_h)
        out = merge_heads(out_h, self.config.n_heads)
        out = self.out_proj(out)
        return self.out_dropout(out)


class FeedForward(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.m_ff_proj = nn.Linear(config.d_model, config.d_ff)
        self.act = activations.NewGELUActivation()
        self.ff_m_proj = nn.Linear(config.d_ff, config.d_model)
        self.ff_dropout = nn.Dropout(config.dropout)

    @typechecked
    def forward(self, x: Float[torch.Tensor, "b l d"]) -> Float[torch.Tensor, "b l d"]:
        x = self.m_ff_proj(x)
        x = self.act(x)
        x = self.ff_m_proj(x)
        return self.ff_dropout(x)


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.attn = SelfAttention(config)
        self.norm_attn = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)
        self.norm_ffn = nn.LayerNorm(config.d_model)

    @typechecked
    def forward(self, x: Float[torch.Tensor, "b l d"]) -> Float[torch.Tensor, "b l d"]:
        x = self.attn(self.norm_attn(x)) + x

        x = self.ffn(self.norm_ffn(x)) + x
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.d_vocab, config.d_model)
        self.positional_encoding = nn.Embedding(config.max_size, config.d_model)

        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.norm_transformer = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.d_vocab, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_normal_(module.weight)
        elif isinstance(module, (nn.LayerNorm,)):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @typechecked
    def forward(self, x: Int[torch.Tensor, "b l"]) -> Float[torch.Tensor, "b l d"]:
        x = self.embedding(x) + self.positional_encoding(
            torch.arange(x.size(1), device=x.device, dtype=torch.int64).unsqueeze(0)
        )
        for layer in self.layers:
            x = layer(x)

        x = self.norm_transformer(x)
        return self.lm_head(x)

    @typechecked
    def generate(
        self,
        x: Int[torch.Tensor, "b l"],
        n_tokens: int = 100,
        n_beams: int = 1,
        strategy: GenerationStrategies = GenerationStrategies.DETERMINIST,
        temperature: float = 0.0,
    ) -> Int[torch.Tensor, "b ll"]:
        match strategy:
            case GenerationStrategies.DETERMINIST:
                assert (
                    temperature == 0.0
                ), f"{strategy = } and {temperature = } are incompatible, temperature must be 0.0"
                return generate_greedy(self, x, n_tokens, temperature=0.0)
            case GenerationStrategies.SAMPLING:
                return generate_greedy(self, x, n_tokens, temperature=temperature)
            case GenerationStrategies.BEAM_SEARCH:
                return generate_beam_search(
                    self, x, n_tokens, n_beams, temperature=temperature
                )
            case _:
                raise NotImplementedError()

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
    ):
        config_hf = AutoConfig.from_pretrained(model_name)

        config = Config(
            d_model=config_hf.hidden_size,
            d_vocab=config_hf.vocab_size,
            max_size=config_hf.n_positions,
            d_ff=4 * config_hf.hidden_size,
            n_heads=config_hf.n_head,
            n_layers=config_hf.n_layer,
            is_causal=True,
            dropout=config_hf.attn_pdrop,
        )

        gpt2_layer_translation = {
            "transformer.": "",
            "wte": "embedding",
            "wpe": "positional_encoding",
            "h.": "layers.",
            "ln_1": "norm_attn",
            "ln_2": "norm_ffn",
            "attn.c_attn": "attn.qkv_proj",
            "attn.c_proj": "attn.out_proj",
            "mlp.c_fc": "ffn.m_ff_proj",
            "mlp.c_proj": "ffn.ff_m_proj",
            "ln_f": "norm_transformer",
        }

        model_hf = AutoModelForCausalLM.from_pretrained(model_name)
        sd_hf = model_hf.state_dict()

        model = cls(config)
        sd = model.state_dict()

        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        copied_layers = set()
        for k_hf, v_hf in sd_hf.items():
            k = k_hf
            for tr_hf, tr in gpt2_layer_translation.items():
                if tr_hf in k:
                    k = k.replace(tr_hf, tr)
            if any(t in k_hf for t in transposed):
                v_hf = v_hf.transpose(0, 1)
            assert k in sd, f"{k = } not in {sd.keys()}"
            assert sd[k].shape == v_hf.shape, f"{sd[k].shape = } != {v_hf.shape = }"

            with torch.no_grad():
                sd[k].copy_(v_hf)
            copied_layers.add(k)

        return model.eval()
