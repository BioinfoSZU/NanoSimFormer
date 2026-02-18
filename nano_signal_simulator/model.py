import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.parametrizations import weight_norm
import torchtune
import torchtune.models.llama3_1
from flash_attn import flash_attn_qkvpacked_func
from flash_attn.layers.rotary import RotaryEmbedding
from flash_attn.modules.mlp import GatedMlp
from flash_attn.ops.triton.layer_norm import RMSNorm
from functools import lru_cache
from typing import Optional


class Transpose(torch.nn.Module):
    def __init__(self, shape: tuple):
        super().__init__()
        self.shape = shape

    def forward(self, inputs: Tensor):
        return inputs.transpose(*self.shape)


def deepnorm_params(depth):
    alpha = round((2*depth)**0.25, 7)
    beta = round((8*depth)**(-1/4), 7)
    return alpha, beta


@lru_cache(maxsize=2)
def sliding_window_mask(seq_len, window, device):
    band = torch.full((seq_len, seq_len), fill_value=1.0)
    band = torch.triu(band, diagonal=-window[0])
    band = band * torch.tril(band, diagonal=window[1])
    band = band.to(torch.bool).to(device)
    return band


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.Wqkv = torch.nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = torch.nn.Linear(d_model, d_model, bias=True)
        self.rotary_emb = RotaryEmbedding(self.head_dim, interleaved=False)

    @staticmethod
    def attn_func(qkv):
        if torch.cuda.get_device_capability(qkv.device)[0] >= 8 and (torch.is_autocast_enabled() or qkv.dtype == torch.half):
            attn_output = flash_attn_qkvpacked_func(qkv, causal=False, window_size=(127, 128))
        else:
            q, k, v = torch.chunk(qkv.permute(0, 2, 3, 1, 4), chunks=3, dim=1)
            mask = sliding_window_mask(qkv.shape[1], (127, 128), q.device)
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            attn_output = attn_output.permute(0, 1, 3, 2, 4)
        return attn_output

    def forward(self, x):
        N, T, _ = x.shape
        qkv = self.Wqkv(x).view(N, T, 3, self.nhead, self.head_dim)
        qkv = self.rotary_emb(qkv)
        attn_output = MultiHeadAttention.attn_func(qkv).reshape(N, T, self.d_model)
        out = self.out_proj(attn_output)
        return out


class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, deepnorm_alpha: float, deepnorm_beta: float):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward

        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            nhead=nhead,
        )
        self.ff = GatedMlp(
            d_model,
            hidden_features=dim_feedforward,
            activation=F.silu,
            bias1=False,
            bias2=False,
            multiple_of=1,
        )
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.register_buffer("deepnorm_alpha", torch.tensor(deepnorm_alpha))
        self.deepnorm_beta = deepnorm_beta
        self.reset_parameters()

    def reset_parameters(self):
        db = self.deepnorm_beta
        d_model = self.d_model
        torch.nn.init.xavier_normal_(self.ff.fc1.weight, gain=db)
        torch.nn.init.xavier_normal_(self.ff.fc2.weight, gain=db)
        torch.nn.init.xavier_normal_(self.self_attn.out_proj.weight, gain=db)
        torch.nn.init.xavier_normal_(self.self_attn.Wqkv.weight[2*d_model:], gain=db)
        torch.nn.init.xavier_normal_(self.self_attn.Wqkv.weight[:2*d_model], gain=1)

    def forward(self, x):
        x = self.norm1(self.self_attn(x), self.deepnorm_alpha * x)
        x = self.norm2(self.ff(x), self.deepnorm_alpha * x)
        return x


class ConvolutionModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, padding_mode='reflect'):
        super().__init__()
        self.conv = weight_norm(nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            padding_mode=padding_mode,
        ))

    def forward(self, x):
        return self.conv(x)


class TransposeConvolutionModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias):
        super().__init__()
        self.conv = weight_norm(nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        ))

    def forward(self, x):
        return self.conv(x)


class ConvolutionResnetBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, compress=2, padding_mode='reflect'):
        super().__init__()
        hidden_dim = dim // compress
        self.conv_modules = nn.Sequential(
            nn.SiLU(),
            ConvolutionModule(
                in_channels=dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size//2,
                bias=True,
                padding_mode=padding_mode,
            ),
            nn.SiLU(),
            ConvolutionModule(
                in_channels=hidden_dim,
                out_channels=dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
                padding_mode=padding_mode,
            )
        )

    def forward(self, x):
        return x + self.conv_modules(x)


class SignalDecoder(nn.Module):
    def __init__(self, dim=512, output_dim=1, num_transformer_layers=8, padding_mode='reflect'):
        super().__init__()
        self.output_dim = output_dim

        alpha, beta = deepnorm_params(num_transformer_layers)
        self.transformer_layers = torch.nn.ModuleList([
            TransformerEncoderLayer(
                d_model=dim,
                nhead=8,
                dim_feedforward=2048,
                deepnorm_alpha=alpha,
                deepnorm_beta=beta,
            )
            for _ in range(num_transformer_layers)
        ])

        self.conv_layers = nn.ModuleList([
            Transpose(shape=(1, 2)),
            TransposeConvolutionModule(
                in_channels=dim, out_channels=256, kernel_size=7, stride=1, padding=3, output_padding=0, bias=True,
            ),
            ConvolutionResnetBlock(dim=256, padding_mode=padding_mode),
            nn.SiLU(),

            TransposeConvolutionModule(
                in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=0, bias=True,
            ),
            ConvolutionResnetBlock(dim=128, padding_mode=padding_mode),
            nn.SiLU(),

            TransposeConvolutionModule(
                in_channels=128, out_channels=64, kernel_size=6, stride=3, padding=2, output_padding=0, bias=True,
            ),
            ConvolutionResnetBlock(dim=64, padding_mode=padding_mode),
            nn.SiLU(),

            ConvolutionModule(in_channels=64, out_channels=output_dim, kernel_size=5, stride=1, padding=2, bias=True, padding_mode=padding_mode),
            Transpose(shape=(1, 2)),
        ])

    def forward(self, x):
        for layer in self.transformer_layers:
            x = layer(x)
        for layer in self.conv_layers:
            x = layer(x)
        if self.output_dim == 1:
            x = x.squeeze(-1)
        return x


class RNASignalDecoder(nn.Module):
    def __init__(self, dim=512, output_dim=1, num_transformer_layers=8, padding_mode='reflect'):
        super().__init__()
        self.output_dim = output_dim

        alpha, beta = deepnorm_params(num_transformer_layers)
        self.transformer_layers = torch.nn.ModuleList([
            TransformerEncoderLayer(
                d_model=dim,
                nhead=8,
                dim_feedforward=2048,
                deepnorm_alpha=alpha,
                deepnorm_beta=beta,
            )
            for _ in range(num_transformer_layers)
        ])

        self.conv_layers = nn.ModuleList([
            Transpose(shape=(1, 2)),
            TransposeConvolutionModule(
                in_channels=dim, out_channels=256, kernel_size=7, stride=1, padding=3, output_padding=0, bias=True,
            ),
            ConvolutionResnetBlock(dim=256, padding_mode=padding_mode),
            nn.SiLU(),

            TransposeConvolutionModule(
                in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, output_padding=0, bias=True,
            ),
            ConvolutionResnetBlock(dim=128, padding_mode=padding_mode),
            nn.SiLU(),

            TransposeConvolutionModule(
                in_channels=128, out_channels=64, kernel_size=7, stride=3, padding=2, output_padding=0, bias=True,
            ),
            ConvolutionResnetBlock(dim=64, padding_mode=padding_mode),
            nn.SiLU(),

            ConvolutionModule(in_channels=64, out_channels=output_dim, kernel_size=5, stride=1, padding=2, bias=True, padding_mode=padding_mode),
            Transpose(shape=(1, 2)),
        ])

    def forward(self, x):
        for layer in self.transformer_layers:
            x = layer(x)
        for layer in self.conv_layers:
            x = layer(x)
        if self.output_dim == 1:
            x = x.squeeze(-1)
        return x


class SequenceEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        padding_idx: Optional[int] = 0,
        decoder_dim=512,
        decoder_max_seq_len=500,
        num_layers=12,
        norm_eps=1e-05,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=decoder_dim, padding_idx=padding_idx)

        self.decoder_max_seq_len = decoder_max_seq_len
        self.num_layers = num_layers
        self.decoder_layers = nn.ModuleList([
            torchtune.modules.TransformerSelfAttentionLayer(
                attn=torchtune.modules.MultiHeadAttention(
                    embed_dim=decoder_dim,
                    num_heads=8,
                    num_kv_heads=8,
                    head_dim=64,
                    q_proj=nn.Linear(decoder_dim, decoder_dim, bias=False),
                    k_proj=nn.Linear(decoder_dim, decoder_dim, bias=False),
                    v_proj=nn.Linear(decoder_dim, decoder_dim, bias=False),
                    output_proj=nn.Linear(decoder_dim, decoder_dim, bias=False),
                    pos_embeddings=torchtune.models.llama3_1._position_embeddings.Llama3ScaledRoPE(
                        64, max_seq_len=decoder_max_seq_len, base=10000, scale_factor=8,
                    ),
                    q_norm=None,
                    k_norm=None,
                    max_seq_len=decoder_max_seq_len,
                    is_causal=False,
                    attn_dropout=0.0,
                ),
                mlp=torchtune.modules.FeedForward(
                    gate_proj=nn.Linear(decoder_dim, 2048),
                    down_proj=nn.Linear(2048, decoder_dim),
                    up_proj=nn.Linear(decoder_dim, 2048),
                ),
                sa_norm=torchtune.modules.RMSNorm(decoder_dim, eps=norm_eps),
                mlp_norm=torchtune.modules.RMSNorm(decoder_dim, eps=norm_eps),
            )
            for _ in range(num_layers)
        ])
        self.final_norm = torchtune.modules.RMSNorm(decoder_dim, eps=norm_eps)
        self.output_layer = nn.Linear(decoder_dim, decoder_dim)

    def forward(self, targets, mask):
        h = self.tok_embeddings(targets)
        for i, layer in enumerate(self.decoder_layers):
            h = layer(
                h,
                mask=mask,
            )
        h = self.output_layer(self.final_norm(h)).contiguous()
        return h


class Model(nn.Module):
    def __init__(
        self,
        dim,
        m_type,
    ):
        super().__init__()

        self.seq_encoder = SequenceEncoder(
            vocab_size=5,
            padding_idx=0,
            decoder_dim=dim,
            decoder_max_seq_len=1024,
            num_layers=12,
        )

        if m_type == "DNA":
            self.sig_decoder = SignalDecoder(
                dim=dim,
                num_transformer_layers=8,
                output_dim=1,
                padding_mode='reflect',
            )
        elif m_type == "RNA":
            self.sig_decoder = RNASignalDecoder(
                dim=dim,
                num_transformer_layers=8,
                output_dim=1,
                padding_mode='reflect',
            )
        else:
            raise NotImplementedError
