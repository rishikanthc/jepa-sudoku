import math
from dataclasses import dataclass

import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor

from ssp import ThreeAxisSSP, ThreeAxisSSPConfig


@dataclass
class TransformerConfig:
    context_size: int = 81
    n_heads: int = 4
    head_dim: int = 128
    n_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1

    @property
    def d_model(self) -> int:
        return self.n_heads * self.head_dim


class SA(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        self.drop1 = nn.Dropout(config.dropout)
        self.drop2 = nn.Dropout(config.dropout)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

    def forward(self, x: Float[Tensor, "b s d"]) -> Float[Tensor, "b s d"]:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q, "b s (h d) -> b h s d", h=self.n_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.n_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.n_heads)

        attn_scores = einsum(q, k, "b h s_q d, b h s_k d -> b h s_q s_k") / math.sqrt(
            self.head_dim
        )
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.drop1(attn_probs)

        out = einsum(v, attn_probs, "b h s_v d, b h s_v s_k -> b h s_k d")
        out = rearrange(out, "b h s d -> b s (h d)")
        out = self.out_proj(out)
        out = self.drop2(out)

        return out


class CA(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(self.d_model, self.d_model, False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, False)

        self.drop1 = nn.Dropout(config.dropout)
        self.drop2 = nn.Dropout(config.dropout)

        self.out_proj = nn.Linear(self.d_model, self.d_model, False)

    def forward(
        self, x: Float[Tensor, "b s d"], s2: Float[Tensor, "b t d"]
    ) -> Float[Tensor, "b s d"]:
        q = self.q_proj(x)
        k = self.k_proj(s2)
        v = self.v_proj(s2)

        q = rearrange(q, "b s (h d) -> b h s d", h=self.n_heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.n_heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.n_heads)

        attn_scores = einsum(q, k, "b h s d, b h t d -> b h s t") / math.sqrt(
            self.head_dim
        )
        attn_prob = F.softmax(attn_scores, dim=-1)
        attn_prob = self.drop1(attn_prob)

        out = einsum(v, attn_prob, "b h t d, b h s t -> b h s d")
        out = rearrange(out, "b h s d -> b s (h d)")
        out = self.out_proj(out)
        out = self.drop2(out)

        return out


class FFN(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Float[Tensor, "b s d"]) -> Float[Tensor, "b s d"]:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig, cross_attn: bool = False):
        super().__init__()

        self.cross_attn = cross_attn
        self.self_attention = SA(config)
        self.ln_sa = nn.LayerNorm(config.d_model)
        self.ln_ffn = nn.LayerNorm(config.d_model)

        if cross_attn:
            self.cross_attention = CA(config)
            self.ln_ca = nn.LayerNorm(config.d_model)

        self.ffn = FFN(config)

    def forward(
        self,
        x: Float[Tensor, "b s d"],
        encoder_out: None | Float[Tensor, "b t d"] = None,
    ) -> Float[Tensor, "b s d"]:
        out = x + self.self_attention(self.ln_sa(x))
        out = x + out

        if self.cross_attn:
            if encoder_out is None:
                raise ValueError("cross_attention requires encoder_out")
            out = out + self.cross_attention(self.ln_ca(out), encoder_out)

        out = out + self.ffn(self.ln_ffn(out))

        return out


class Encoder(nn.Module):
    def __init__(self, config: TransformerConfig, embedding_config: ThreeAxisSSPConfig):
        super().__init__()

        self.config = config
        self.embedding = ThreeAxisSSP(embedding_config)

        # self.embed = <SSP encoder>

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.drop = nn.Dropout(config.dropout)
        self.out_ln = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.d_model, False)

    def forward(self, x: Float[Tensor, "b 81 3"]) -> Float[Tensor, "b 81 d"]:
        tokens = self.embedding(x)
        out = self.drop(tokens)

        for block in self.blocks:
            out = block(out)

        out = self.out_ln(out)
        out = self.head(out)

        return out


class Predictor(nn.Module):
    def __init__(self, config: TransformerConfig, embedding_config: ThreeAxisSSPConfig):
        super().__init__()

        self.config = config
        self.embedding = ThreeAxisSSP(embedding_config)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config, True) for _ in range(config.n_layers)]
        )
        self.drop = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.d_model, config.d_model, False)
        self.out_ln = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.d_model, False)

    def forward(
        self, x: Float[Tensor, "b t 3"], encoder_out: Float[Tensor, "b 81 d"]
    ) -> Float[Tensor, "b t d"]:
        tokens = self.embedding(x)
        out = self.drop(tokens)

        for block in self.blocks:
            out = block(out, encoder_out)

        out = self.out_ln(out)
        out = self.head(out)

        return out
