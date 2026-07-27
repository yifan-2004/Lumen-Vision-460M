"""VisionPsyNano vision tower + modality projector, ported verbatim from the reference
implementation (`models/{vision_transformer,modality_projector}.py`).

Why port instead of reusing vLLM's SiglipVisionModel:
  * The vision encoder is a small, fixed-length (1024-patch) bidirectional pass --
    it does NOT need paged attention / KV cache, so a plain nn.Module is fine and
    runs batched across tiles inside the vLLM model.
  * Keeping the exact submodule names (`patch_embedding.conv`, `blocks.N.attn.qkv_proj`,
    `blocks.N.ln1`, ...) lets the original `vision_encoder.*` checkpoint weights load
    1:1 with no remap, which guarantees numerical parity with the reference model.

These modules carry NO vLLM dependency on purpose.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTPatchEmbeddings(nn.Module):
    def __init__(self, img_size, patch_size, hidden_dim, cls_flag):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.cls_flag = cls_flag
        self.embd_dim = hidden_dim
        self.conv = nn.Conv2d(
            in_channels=3, out_channels=hidden_dim,
            kernel_size=patch_size, stride=patch_size, padding="valid",
        )
        if self.cls_flag:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.position_embedding = nn.Parameter(torch.rand(1, self.num_patches + 1, hidden_dim))
        else:
            self.position_embedding = nn.Parameter(torch.rand(1, self.num_patches, hidden_dim))

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        if self.cls_flag:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        x = x + self.position_embedding
        return x


class ViTMultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.embd_dim = hidden_dim
        assert hidden_dim % n_heads == 0, "embd_dim must be divisible by num_heads"
        self.head_dim = hidden_dim // n_heads
        self.dropout = dropout
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        y = self.resid_dropout(y)
        return y


class ViTMLP(nn.Module):
    def __init__(self, hidden_dim, inter_dim, dropout):
        super().__init__()
        self.activation_fn = nn.GELU(approximate='tanh')
        self.fc1 = nn.Linear(hidden_dim, inter_dim)
        self.fc2 = nn.Linear(inter_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class ViTBlock(nn.Module):
    def __init__(self, hidden_dim, inter_dim, n_heads, dropout, ln_eps):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim, eps=ln_eps)
        self.attn = ViTMultiHeadAttention(hidden_dim, n_heads, dropout)
        self.ln2 = nn.LayerNorm(hidden_dim, eps=ln_eps)
        self.mlp = ViTMLP(hidden_dim, inter_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ViT(nn.Module):
    """SigLIP-style vision encoder. Matches the reference `ViT` submodule names so the
    `vision_encoder.*` checkpoint tensors load 1:1."""

    def __init__(self, vit_hidden_dim, vit_inter_dim, vit_patch_size, vit_img_size,
                 vit_n_heads, vit_n_blocks, vit_dropout, vit_ln_eps, vit_cls_flag):
        super().__init__()
        self.patch_embedding = ViTPatchEmbeddings(
            vit_img_size, vit_patch_size, vit_hidden_dim, vit_cls_flag)
        self.cls_flag = vit_cls_flag
        self.dropout = nn.Dropout(vit_dropout)
        self.blocks = nn.ModuleList([
            ViTBlock(vit_hidden_dim, vit_inter_dim, vit_n_heads, vit_dropout, vit_ln_eps)
            for _ in range(vit_n_blocks)
        ])
        self.layer_norm = nn.LayerNorm(vit_hidden_dim, eps=vit_ln_eps)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        if self.cls_flag:
            x = self.layer_norm(x[:, 0])
        else:
            x = self.layer_norm(x)
        return x


class ModalityProjector(nn.Module):
    """pixel-shuffle (factor s) + Linear. Matches the reference `ModalityProjector`
    (`MP.proj.weight`)."""

    def __init__(self, vit_hidden_dim, lm_hidden_dim, pixel_shuffle_factor):
        super().__init__()
        self.input_dim = vit_hidden_dim * (pixel_shuffle_factor ** 2)
        self.output_dim = lm_hidden_dim
        self.scale_factor = pixel_shuffle_factor
        self.proj = nn.Linear(self.input_dim, self.output_dim, bias=False)

    def pixel_shuffle(self, x):
        bsz, seq, embed_dim = x.size()
        seq_root = int(seq ** 0.5)
        assert seq_root ** 2 == seq
        assert seq_root % self.scale_factor == 0
        height = width = seq_root
        x = x.view(bsz, height, width, embed_dim)
        h_out = height // self.scale_factor
        w_out = width // self.scale_factor
        x = x.reshape(bsz, h_out, self.scale_factor, w_out, self.scale_factor, embed_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(bsz, h_out * w_out, embed_dim * self.scale_factor ** 2)
        return x

    def forward(self, x):
        x = self.pixel_shuffle(x)
        x = self.proj(x)
        return x
