from dataclasses import dataclass
from typing import Optional, Union, List

import torch
import numpy as np
import math
from torch import Tensor, nn
from einops import rearrange

from ..modules.layers import (DoubleStreamBlock, EmbedND, LastLayer,
                                 MLPEmbedder, SingleStreamBlock,
                                 timestep_embedding)
from einops import repeat

# -------------------------------------------------------------------------
# 1) Flux model
# -------------------------------------------------------------------------
@dataclass
class FluxParams:
    in_channels: int
    out_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """
    _supports_gradient_checkpointing = True

    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels if params.out_channels is not None else self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)

        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,)
                for i in range(1, params.depth+1)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size, 
                    self.num_heads, 
                    mlp_ratio=params.mlp_ratio,
                    )
                for i in range(1, params.depth_single_blocks+1)
            ]
        )
       
        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)
        self.gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    
    @property
    def attn_processors(self):
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        txt_vec: Tensor,
        block_controlnet_hidden_states=None,
        guidance: Tensor = None,
        image_proj: Tensor = None, 
        ip_scale: Tensor = 1.0, 
        return_intermediate: bool = False,
        cxt_embeddings: Tensor = None,
        task_register_embeddings: Tensor = None,
        use_flash_attention: bool = False,
    ) -> Tensor:
            
        if return_intermediate:
            intermediate_double = []
            intermediate_single = []
            
        # running on sequences img
        img = self.img_in(img)

        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(txt_vec)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        if block_controlnet_hidden_states is not None:
            controlnet_depth = len(block_controlnet_hidden_states)

        for index_block, block in enumerate(self.double_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                img, txt = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    img,
                    txt,
                    vec,
                    pe,
                    cxt_embeddings,
                    task_register_embeddings,
                    image_proj,
                    ip_scale,
                    use_flash_attention,
                )
            else:
                img, txt = block(
                    img=img, 
                    txt=txt, 
                    vec=vec, 
                    pe=pe,
                    cxt_embeddings=cxt_embeddings,
                    task_register_embeddings=task_register_embeddings,
                    image_proj=image_proj,
                    ip_scale=ip_scale,
                    use_flash_attention=use_flash_attention,
                )



            if return_intermediate:
                intermediate_double.append(
                    [img, txt]
                )

            if block_controlnet_hidden_states is not None:
                img = img + block_controlnet_hidden_states[index_block % 2]

        img = torch.cat((txt, img), dim=1)
        for index_block, block in enumerate(self.single_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                img = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    img,
                    vec,
                    pe,
                    cxt_embeddings,
                    task_register_embeddings,
                    image_proj,
                    ip_scale,
                    use_flash_attention,
                )
            else:
                img = block(
                    img=img, 
                    vec=vec, 
                    pe=pe, 
                    cxt_embeddings=cxt_embeddings, 
                    task_register_embeddings=task_register_embeddings, 
                    image_proj=image_proj, 
                    ip_scale=ip_scale,
                    use_flash_attention=use_flash_attention,
                )

            if return_intermediate:
                intermediate_single.append([
                    img[:, txt.shape[1]:, ...],
                    img[:, :txt.shape[1], ...]
                ])

        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        if return_intermediate:
            return img, intermediate_double, intermediate_single
        else:
            return img


# -------------------------------------------------------------------------
# 2) IC-Custom model
# -------------------------------------------------------------------------
class IC_Custom(Flux):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cond_embedding = nn.Embedding(1, 3072)
        self.cond_embedding.weight.data.zero_()
        self.target_embedding = nn.Embedding(1, 3072)
        self.target_embedding.weight.data.zero_()
        self.idx_embedding = nn.Embedding(10, 3072)
        self.idx_embedding.weight.data.zero_()

        self.task_register_embeddings = nn.Embedding(24, 3072)
        self.task_register_embeddings.weight.data.zero_()


    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        txt_vec: Tensor,
        guidance: Tensor = None,
        return_intermediate: bool = False,
        cond_w_regions: Optional[Union[List[int], int]] = None,
        mask_type_ids: Optional[Union[Tensor, int]] = None,
        height: int = 1024,
        width: int = 1024,
        use_flash_attention: bool = False,
    ) -> Tensor:

        batch_size, seq_len, _ = img.shape

        ## 1. calculate the sequence length
        # Units: 2x2 latent tokens (VAE downscale=8, pack=2)
        latent_h = math.ceil(height // 8 // 2)
        latent_w = math.ceil(width  // 8 // 2)
        assert latent_h * latent_w == seq_len

        if isinstance(cond_w_regions, list):
            cond_w_list = [math.ceil(cond_w // 8 // 2) for cond_w in cond_w_regions]
        else:
            cond_w_list = [math.ceil(cond_w_regions // 8 // 2)]

        cond_w_total = sum(cond_w_list)
        tgt_w = latent_w - cond_w_total


        ## 2. calculate the embeddings and rearrange them
        cond_embeddings = repeat(self.cond_embedding.weight[0], "c -> n l c", n=batch_size, l=cond_w_total * latent_h)
        cond_embeddings = rearrange(cond_embeddings, "b (h w) d -> b h w d", h=latent_h, w=cond_w_total)

        target_embeddings = repeat(self.target_embedding.weight[0], "c -> n l c", n=batch_size, l=latent_h*tgt_w)
        target_embeddings = rearrange(target_embeddings, "b (h w) d -> b h w d", h=latent_h, w=tgt_w)

        idx_embeddings = []
        for idx in range(len(cond_w_list)):
            idx_embedding  = repeat(self.idx_embedding.weight[idx], "c -> n h w c", n=batch_size, h=latent_h, w=cond_w_list[idx])
            idx_embeddings.append(idx_embedding)
        
        ## 3. combine the embeddings
        idx_embeddings = torch.cat(idx_embeddings, dim=2)
        cond_embeddings = cond_embeddings + idx_embeddings

        cxt_embeddings = torch.cat((cond_embeddings, target_embeddings), dim=2)
        cxt_embeddings = rearrange(cxt_embeddings, "b h w d -> b (h w) d")

        ## 4. calculate the mask type embeddings
        if isinstance(mask_type_ids, int):
            mask_type_ids = torch.tensor([mask_type_ids], device=img.device, dtype=img.dtype)
        task_register_embeddings = self.task_register_embeddings.weight.view(3, 2, 4, -1)[mask_type_ids.long()] ## num_mask_types, k&v, num_tokens, d

        ## 5. forward the model
        return super().forward(
            img=img, 
            img_ids=img_ids, 
            txt=txt, 
            txt_ids=txt_ids, 
            timesteps=timesteps, 
            txt_vec=txt_vec, 
            guidance=guidance, 
            return_intermediate=return_intermediate, 
            cxt_embeddings=cxt_embeddings, 
            task_register_embeddings=task_register_embeddings,
            use_flash_attention=use_flash_attention
            )
