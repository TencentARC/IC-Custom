import math
from dataclasses import dataclass

import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from .math import attention, rope, apply_learnable_pos_emb


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor):
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, network_alpha=None, device=None, dtype=None):
        super().__init__()

        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank
        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)


class FLuxSelfAttnProcessor:
    def __call__(self, attn, x, pe, use_flash_attention: bool = False, **attention_kwargs):
        print('2' * 30)

        qkv = attn.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = attn.norm(q, k, v)
        x = attention(q, k, v, pe=pe, use_flash_attention=use_flash_attention)
        x = attn.proj(x)
        return x


class LoraFluxAttnProcessor(nn.Module):

    def __init__(self, dim: int, rank=4, network_alpha=None, lora_weight=1):
        super().__init__()
        self.qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.lora_weight = lora_weight


    def __call__(self, attn, x, pe, use_flash_attention: bool = False, **attention_kwargs):
        qkv = attn.qkv(x) + self.qkv_lora(x) * self.lora_weight
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = attn.norm(q, k, v)
        x = attention(q, k, v, pe=pe, use_flash_attention=use_flash_attention)
        x = attn.proj(x) + self.proj_lora(x) * self.lora_weight
        print('1' * 30)
        print(x.norm(), (self.proj_lora(x) * self.lora_weight).norm(), 'norm')
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, context: bool = False, zero_out: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.kv_context = nn.Linear(dim, dim * 2, bias=qkv_bias) if context else None
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim) if not zero_out else zero_module(nn.Linear(dim, dim))
    def forward():
        pass
    

class ContextAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.norm = QKNorm(head_dim)
        # self.proj = nn.Linear(dim, dim)
        self.proj = zero_module(nn.Linear(dim, dim))
        
    def forward():
        pass


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor):
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlockLoraProcessor(nn.Module):
    def __init__(self, dim: int, rank=4, network_alpha=None, lora_weight=1):
        super().__init__()
        self.qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.qkv_lora2 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora2 = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.lora_weight = lora_weight

    def __call__(self, attn, img, txt, vec, pe, cxt_embeddings=None, task_register_embeddings: Tensor = None, image_proj: Tensor = None, ip_scale: float = 1.0, use_flash_attention: bool = False, **attention_kwargs):


        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated) + self.qkv_lora1(img_modulated) * self.lora_weight
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated) + self.qkv_lora2(txt_modulated) * self.lora_weight
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        if cxt_embeddings is not None:
            cxt_embeddings = rearrange(cxt_embeddings, "B L (H D) -> B H L D", H=attn.num_heads)
            img_q, img_k = apply_learnable_pos_emb(img_q, img_k, cxt_embeddings)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        if task_register_embeddings is not None:
            task_register_embeddings = rearrange(task_register_embeddings, "B K L (H D) -> K B H L D", K=2, H=attn.num_heads)
            attn1 = attention(q, k, v, pe=pe, task_register_embeddings=task_register_embeddings, use_flash_attention=use_flash_attention)
        else:
            attn1 = attention(q, k, v, pe=pe, use_flash_attention=use_flash_attention)

        txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn) + img_mod1.gate * self.proj_lora1(img_attn) * self.lora_weight
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn) + txt_mod1.gate * self.proj_lora2(txt_attn) * self.lora_weight
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        return img, txt


class DoubleStreamBlockProcessor(nn.Module):
    def __init__(self, dim=-1, load_lora=False, rank=4, network_alpha=None, lora_weight=1):
        super().__init__()
        self.load_lora = load_lora
        if load_lora:
            self.qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
            self.proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)
            self.qkv_lora2 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
            self.proj_lora2 = LoRALinearLayer(dim, dim, rank, network_alpha)
            self.lora_weight = lora_weight

    def __call__(self, attn, img, txt, vec, pe, cxt_embeddings=None, task_register_embeddings: Tensor = None, image_proj: Tensor = None, ip_scale: float = 1.0, use_flash_attention: bool = False, **attention_kwargs):
        self.to(img.device)
     
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        
        # prepare image for attention
        img_modulated = attn.img_norm1(img)

        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        if self.load_lora:
            img_qkv = img_qkv + self.qkv_lora1(img_modulated) * self.lora_weight
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)
    

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        if self.load_lora:
            txt_qkv = txt_qkv + self.qkv_lora2(txt_modulated) * self.lora_weight
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        if cxt_embeddings is not None:
            cxt_embeddings = rearrange(cxt_embeddings, "B L (H D) -> B H L D", H=attn.num_heads)
            img_q, img_k = apply_learnable_pos_emb(img_q, img_k, cxt_embeddings)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        if task_register_embeddings is not None:
            task_register_embeddings = rearrange(task_register_embeddings, "B K L (H D) -> K B H L D", K=2, H=attn.num_heads)
            attn1 = attention(q, k, v, pe=pe, task_register_embeddings=task_register_embeddings, use_flash_attention=use_flash_attention)
        else:
            attn1 = attention(q, k, v, pe=pe, use_flash_attention=use_flash_attention)
   
        txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)
        if self.load_lora:
            img = img  +  img_mod1.gate * self.proj_lora1(img_attn) * self.lora_weight

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        if self.load_lora:
            txt = txt  +  txt_mod1.gate * self.proj_lora2(txt_attn) * self.lora_weight
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        
        return img, txt


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )


        processor = DoubleStreamBlockProcessor()
        self.set_processor(processor)

    def set_processor(self, processor) -> None:
        self.processor = processor

    def get_processor(self):
        return self.processor

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        vec: Tensor,
        pe: Tensor,
        cxt_embeddings: Tensor = None,
        task_register_embeddings: Tensor = None,
        image_proj: Tensor = None,
        ip_scale: float =1.0,
        use_flash_attention: bool = False,
    ):
        if image_proj is None:
            return self.processor(self, img, txt, vec, pe, cxt_embeddings, task_register_embeddings, use_flash_attention)
        else:
            return self.processor(self, img, txt, vec, pe, cxt_embeddings, task_register_embeddings, image_proj, ip_scale, use_flash_attention)


class SingleStreamBlockLoraProcessor(nn.Module):
    def __init__(self, dim: int, rank: int = 4, network_alpha = None, lora_weight: float = 1):
        super().__init__()
        self.qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.lora_weight = lora_weight

    def __call__(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor, cxt_embeddings: Tensor = None, task_register_embeddings: Tensor = None, image_proj: Tensor = None, ip_scale: float = 1.0, use_flash_attention: bool = False) -> Tensor:

        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
        qkv = qkv + self.qkv_lora(x_mod) * self.lora_weight

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        q, k = attn.norm(q, k, v)

        # compute attention
        if cxt_embeddings is not None:
            txt_len = q.shape[2] - cxt_embeddings.shape[1]
            txt_q, img_q = q[:, :, : txt_len], q[:, :, txt_len :]
            txt_k, img_k = k[:, :, : txt_len], k[:, :, txt_len :]
            cxt_embeddings = rearrange(cxt_embeddings, "B L (H D) -> B H L D", H=attn.num_heads)
            img_q, img_k = apply_learnable_pos_emb(img_q, img_k, cxt_embeddings)
            q = torch.cat((txt_q, img_q), dim=2)
            k = torch.cat((txt_k, img_k), dim=2)


        if task_register_embeddings is not None:
            task_register_embeddings = rearrange(task_register_embeddings, "B K L (H D) -> K B H L D", K=2, H=attn.num_heads)
            attn1 = attention(q, k, v, pe=pe, task_register_embeddings=task_register_embeddings, use_flash_attention=use_flash_attention)
        else:
            attn1 = attention(q, k, v, pe=pe, use_flash_attention=use_flash_attention)

        # compute activation in mlp stream, cat again and run second linear layer
        output = attn.linear2(torch.cat((attn1, attn.mlp_act(mlp)), 2))
        output = output + self.proj_lora(output) * self.lora_weight
        output = x + mod.gate * output

        return output


class SingleStreamBlockProcessor(nn.Module):

    def __init__(self, dim=0 , rank: int = 32, network_alpha = None, lora_weight: float = 1, load_lora=False):
        super().__init__()
        self.load_lora = load_lora
        if load_lora :
            self.qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
            self.proj_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
            self.lora_weight = lora_weight

    def __call__(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor, cxt_embeddings: Tensor = None, task_register_embeddings: Tensor = None, image_proj: Tensor = None, ip_scale: float = 1.0, use_flash_attention: bool = False) -> Tensor:
        self.to(x.device)
    
        mod, _ = attn.modulation(vec)
        x_mod = attn.pre_norm(x)

            
        x_mod = (1 + mod.scale) * x_mod + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
        if self.load_lora:
            qkv = qkv + self.qkv_lora(x_mod) * self.lora_weight

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        q, k = attn.norm(q, k, v)

        # compute attention
        if cxt_embeddings is not None:
            txt_len = q.shape[2] - cxt_embeddings.shape[1]
            txt_q, img_q = q[:, :, : txt_len], q[:, :, txt_len :]
            txt_k, img_k = k[:, :, : txt_len], k[:, :, txt_len :]
            cxt_embeddings = rearrange(cxt_embeddings, "B L (H D) -> B H L D", H=attn.num_heads)
            img_q, img_k = apply_learnable_pos_emb(img_q, img_k, cxt_embeddings)
            q = torch.cat((txt_q, img_q), dim=2)
            k = torch.cat((txt_k, img_k), dim=2)


        if task_register_embeddings is not None:
            task_register_embeddings = rearrange(task_register_embeddings, "B K L (H D) -> K B H L D", K=2, H=attn.num_heads)
            attn1 = attention(q, k, v, pe=pe, task_register_embeddings=task_register_embeddings, use_flash_attention=use_flash_attention)
        else:
            attn1 = attention(q, k, v, pe=pe, use_flash_attention=use_flash_attention)

        # compute activation in mlp stream, cat again and run second linear layer
        output = attn.linear2(torch.cat((attn1, attn.mlp_act(mlp)), 2))
        if self.load_lora:
            output = output + self.proj_lora(output) * self.lora_weight
        output = x + mod.gate * output


        return output

  
class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(self.head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)



        processor = SingleStreamBlockProcessor()
        self.set_processor(processor)


    def set_processor(self, processor) -> None:
        self.processor = processor

    def get_processor(self):
        return self.processor

    def forward(
        self,
        img: Tensor,
        vec: Tensor,
        pe: Tensor,
        cxt_embeddings: Tensor = None,
        task_register_embeddings: Tensor = None,
        image_proj: Tensor = None,
        ip_scale: float = 1.0,
        use_flash_attention: bool = False,
    ) -> Tensor:
        if image_proj is None:
            return self.processor(self, img, vec, pe, cxt_embeddings, task_register_embeddings, use_flash_attention)
        else:
            return self.processor(self, img, vec, pe, cxt_embeddings, task_register_embeddings, image_proj, ip_scale, use_flash_attention)


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


class ImageProjModel(torch.nn.Module):
    """Projection Model
    https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/ip_adapter.py#L28
    """

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

