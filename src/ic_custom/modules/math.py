import math
import torch
from einops import rearrange
from torch import Tensor

try: 
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None
    print("Flash Attention 2 is not available. Falling back to PyTorch's native attention implementation.")

# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value=None, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False, return_qk_logits=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    if return_qk_logits:
        return attn_weight
    else:
        return attn_weight @ value


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, task_register_embeddings: Tensor = None, use_flash_attention: bool = False) -> Tensor:
    q, k = apply_rope(q, k, pe)

    if task_register_embeddings is not None:
        k_mask_type_embeds, v_mask_type_embeds = task_register_embeddings[0], task_register_embeddings[1]
        k = torch.cat((k, k_mask_type_embeds), dim=2)
        v = torch.cat((v, v_mask_type_embeds), dim=2)

    # Use Flash Attention 2 for acceleration (if available)
    try:
        if use_flash_attention and flash_attn_func is not None:
            # Check if the number of heads match for GQA (Grouped Query Attention)
            q_heads = q.size(1)
            k_heads = k.size(1)
            v_heads = v.size(1)
            
            # Handle the case where k,v have fewer heads than q (Grouped Query Attention)
            if q_heads % k_heads == 0 and q_heads % v_heads == 0:
                # For GQA, we need to repeat k and v
                k = k.repeat_interleave(q_heads // k_heads, dim=1)
                v = v.repeat_interleave(q_heads // v_heads, dim=1)
            elif k_heads != q_heads or v_heads != q_heads:
                # If heads don't match and can't use GQA, fall back to standard attention
                raise ValueError(f"Incompatible head dimensions: q={q_heads}, k={k_heads}, v={v_heads}")
                
            # Input shape: [B, H, L, D] but flash_attn_func expects [B, L, H, D]
            # Need to transpose dimensions 1 and 2
            q_trans = q.transpose(1, 2)  # [B, L, H, D]
            k_trans = k.transpose(1, 2)  # [B, L, H, D]
            v_trans = v.transpose(1, 2)  # [B, L, H, D]
                        
            # causal=False means non-autoregressive
            x_trans = flash_attn_func(q_trans, k_trans, v_trans, causal=False)  # [B, L, H, D]
            
            # Convert back to original dimension order
            x = x_trans.transpose(1, 2)  # [B, H, L, D]
            
            # Output shape: [B, H, L, D], need to convert to [B, L, H*D]
            x = rearrange(x, "B H L D -> B L (H D)")
        else:
            # Fallback to native implementation
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = rearrange(x, "B H L D -> B L (H D)")
    except Exception as e:
        # Fallback to native implementation if flash attention fails
        print(f"Flash attention failed with error: {e}. Falling back to native implementation.")
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "B H L D -> B L (H D)")
    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq, xk, freqs_cis):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    # print(xq_.shape, freqs_cis.shape)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    if xk is not None:
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
        xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
        return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
    else:
        return xq_out.reshape(*xq.shape).type_as(xq), None


def apply_learnable_pos_emb(q: Tensor, k: Tensor, embeddings: Tensor) -> Tensor:
    if embeddings.ndim == 3:
        embeddings = embeddings.unsqueeze(1)
    q = q + embeddings
    k = k + embeddings
    return q, k
