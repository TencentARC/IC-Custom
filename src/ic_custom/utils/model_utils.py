import os
from dataclasses import dataclass
from typing import Union, Optional

import torch
from huggingface_hub import hf_hub_download


from accelerate.logging import get_logger
from accelerate import state

from safetensors import safe_open
from safetensors.torch import load_file as load_sft
from safetensors.torch import save_file as save_sft

from ..models.model import Flux, FluxParams, IC_Custom
from ..modules.autoencoder import AutoEncoder, AutoEncoderParams
from ..modules.conditioner import HFEmbedder
from ..modules.image_embedders import ReduxImageEncoder

from .process_util import print_load_warning

# Initialize logger with a try-except to handle cases where accelerate state isn't initialized
if state.is_initialized():
    logger = get_logger(__name__, log_level="INFO")
else:
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)


# -------------------------------------------------------------------------
# 1) model definition
# -------------------------------------------------------------------------
DIT_PARAMS = FluxParams(
    in_channels=384,
    out_channels=64,
    vec_in_dim=768,
    context_in_dim=4096,
    hidden_size=3072,
    mlp_ratio=4.0,
    num_heads=24,
    depth=19,
    depth_single_blocks=38,
    axes_dim=[16, 56, 56],
    theta=10_000,
    qkv_bias=True,
    guidance_embed=True,
)



AE_PARAMS = AutoEncoderParams(
    resolution=256,
    in_channels=3,
    ch=128,
    out_ch=3,
    ch_mult=[1, 2, 4, 4],
    num_res_blocks=2,
    z_channels=16,
    scale_factor=0.3611,
    shift_factor=0.1159,
)

@dataclass
class HFModelSpec:
    repo_id: str 
    filename: Optional[str] = None
    ckpt_path: Optional[str] = None


configs = {
    "flux-fill-dev-dit": HFModelSpec(
        repo_id="black-forest-labs/FLUX.1-Fill-dev",
        filename="flux1-fill-dev.safetensors",
        ckpt_path=os.getenv("FLUX_DEV_FILL"),
    ),
    "flux-fill-dev-ae": HFModelSpec(
        repo_id="black-forest-labs/FLUX.1-Fill-dev",
        filename="ae.safetensors",
        ckpt_path=os.getenv("AE"),
    ),
    "t5-v1_1-xxl": HFModelSpec(
        repo_id="google/t5-v1_1-xxl",
        ckpt_path=os.getenv("T5_XXL"),
    ),
    "clip-vit-large-patch14": HFModelSpec(
        repo_id="openai/clip-vit-large-patch14",
        ckpt_path=os.getenv("CLIP_VIT_LARGE_PATCH14"),
    ),
    "siglip-so400m-patch14-384": HFModelSpec(
        repo_id="google/siglip-so400m-patch14-384",
        ckpt_path=os.getenv("SIGLIP_SO400M_PATCH14_384"),
    ),
    "flux1-redux-dev": HFModelSpec(
        repo_id="black-forest-labs/FLUX.1-Redux-dev",
        filename="flux1-redux-dev.safetensors",
        ckpt_path=os.getenv("FLUX1_REDUX_DEV"),
    ),
    "dit_lora_0x1561": HFModelSpec(
        repo_id="TencentARC/IC-Custom",
        filename="dit_lora_0x1561.safetensors",
        ckpt_path=os.getenv("DIT_LORA"),
    ),
    "dit_txt_img_in_0x1561": HFModelSpec(
        repo_id="TencentARC/IC-Custom",
        filename="dit_txt_img_in_0x1561.safetensors",
        ckpt_path=os.getenv("DIT_TXT_IMG_IN"),
    ),
    "dit_boundary_embeddings_0x1561": HFModelSpec(
        repo_id="TencentARC/IC-Custom",
        filename="dit_boundary_embeddings_0x1561.safetensors",
        ckpt_path=os.getenv("DIT_BOUNDARY_EMBEDDINGS"),
    ),
    "dit_task_register_embeddings_0x1561": HFModelSpec(
        repo_id="TencentARC/IC-Custom",
        filename="dit_task_register_embeddings_0x1561.safetensors",
        ckpt_path=os.getenv("DIT_TASK_REGISTER_EMBEDDINGS"),
    ),
}
            


# -------------------------------------------------------------------------
# 2) load model func
# -------------------------------------------------------------------------

def resolve_model_path(
    name: str,
    repo_id_field: str = "repo_id",
    filename_field: str = "filename",
    ckpt_path_field: str = "ckpt_path",
    hf_download: bool = True,
) -> str:
    """
    Resolve a model path from name, handling local paths, config paths, and HF downloads.
    
    Args:
        name: Model name or path
        repo_id_field: Field name in configs for repo_id
        filename_field: Field name in configs for filename (if download needed)
        ckpt_path_field: Field name in configs for checkpoint path
        hf_download: Whether to download from HF if not found locally
        replace_suffix: Whether to replace suffix in filename
        suffix_map: Mapping of suffixes to replace

    Explanation:
        1) Resolve from CLI
        2) Resolve from ENV
        3) Resolve from online HF
        
    Returns:
        Resolved path to the model
    """
    # If it's a direct path, return it
    if os.path.exists(name):
        return name
    
    # Try to get from configs
    if name in configs:
        # Get local path from configs
        path = getattr(configs[name], ckpt_path_field)
        
        # If local path exists, use it
        if os.path.exists(path):
            return path
        
        # If download is allowed and we have repo info
        if (hf_download and 
            hasattr(configs[name], repo_id_field) and 
            getattr(configs[name], repo_id_field) is not None):
            
            # If we need a specific file (not just the repo)
            if filename_field and hasattr(configs[name], filename_field):
                filename = getattr(configs[name], filename_field)
                
                # Download the file
                logger.info(f"Downloading {getattr(configs[name], repo_id_field)}/{filename}")
                return hf_hub_download(
                    getattr(configs[name], repo_id_field),
                    filename,
                )
            
            # If we just need the repo ID
            return getattr(configs[name], repo_id_field)
    
    # If all else fails, assume name is the path/repo_id
    return name


def load_dit(
    name: str,
    device: Union[str, torch.device] = "cuda", 
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Load a Flux model.
    
    Args:
        name: Model name or path
        hf_download: Whether to download from HF if not found locally
        device: Device to load model on
        dtype: Data type for model
        
    Returns:
        model: Loaded Flux model
    """
    # Loading Flux
    logger.info("Initializing Flux model")
    
    # Resolve checkpoint path
    ckpt_path = resolve_model_path(
        name=name,
        repo_id_field="repo_id",
        filename_field="filename",
        ckpt_path_field="ckpt_path",
        hf_download=True,
    )

    # Convert device string to torch.device if needed
    if isinstance(device, str):
        device = torch.device(device)

    # Initialize model
    with device:
        model = Flux(DIT_PARAMS).to(dtype=dtype)
    
    # Load weights
    model = load_model_weights(model, ckpt_path, device=device)
        
    return model


def load_ic_custom(
    name: str,
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Function to load the IC-Custom (FLUX.1-Fill-dev + LoRA weights) model.

    Args:
        name: Model config name or path
        hf_download: Whether to download from HF if not found locally
        device: Device to load model on
        dtype: Data type for model

    Returns:
        model: Loaded IC_Custom model
    """
    logger.info("Initializing IC-Custom model")
    
    # Resolve checkpoint path
   
    ckpt_path = resolve_model_path(
        name=name,
        repo_id_field="repo_id",
        filename_field="filename",
        ckpt_path_field="ckpt_path",
        hf_download=True,
    )
 
    # Convert device string to torch.device if needed
    if isinstance(device, str):
        device = torch.device(device)

    # Initialize model on the specified device
    with device:
        model = IC_Custom(DIT_PARAMS).to(dtype=dtype)

    # Load weights
    model = load_model_weights(model, ckpt_path, device=device)

    return model


def load_embedder(
    name: str,
    is_clip: bool,
    device: Union[str, torch.device],
    max_length: int,
    dtype: torch.dtype,
) -> HFEmbedder:
    """
    Generic function to load an embedder model (T5 or CLIP).
    
    Args:
        name: Model name or path
        is_clip: Whether this is a CLIP model
        device: Device to load model on
        max_length: Maximum sequence length
        dtype: Data type for model
        
    Returns:
        model: Loaded embedder model
    """
    # Convert device string to torch.device if needed
    if isinstance(device, str):
        device = torch.device(device)
    
    # Resolve model path - for embedders we don't need to download specific files,
    # just need the repo_id or local path
    path = resolve_model_path(
        name=name,
        repo_id_field="repo_id",
        filename_field=None,  # No specific file needed
        ckpt_path_field="ckpt_path",
        hf_download=False,  # HFEmbedder handles downloads itself
    )
    
    # Initialize and return the model
    model = HFEmbedder(
        path, 
        max_length=max_length, 
        is_clip=is_clip, 
        torch_dtype=dtype, 
    ).to(device)
    
    return model


def load_t5(
    name: str = "t5-v1_1-xxl", 
    device: Union[str, torch.device] = "cuda", 
    max_length: int = 512, 
    dtype: torch.dtype = torch.bfloat16,
) -> HFEmbedder:
    """
    Load a T5 text encoder model.
    
    Args:
        name: Model name or path
        device: Device to load model on
        max_length: Maximum sequence length
        dtype: Data type for model
        
    Returns:
        model: Loaded T5 model
    """
    logger.info(f"Loading T5 model: {name}")
    return load_embedder(
        name=name,
        is_clip=False,
        device=device,
        max_length=max_length,
        dtype=dtype,
    )


def load_clip(
    name: str = "clip-vit-large-patch14", 
    device: Union[str, torch.device] = "cuda", 
    dtype: torch.dtype = torch.bfloat16,
) -> HFEmbedder:
    """
    Load a CLIP text encoder model.
    
    Args:
        name: Model name or path
        device: Device to load model on
        dtype: Data type for model
        
    Returns:
        model: Loaded CLIP model
    """
    logger.info(f"Loading CLIP model: {name}")
    return load_embedder(
        name=name,
        is_clip=True,
        device=device,
        max_length=77,  # Standard for CLIP
        dtype=dtype,
    )
    

def load_ae(
    name: str, 
    device: Union[str, torch.device] = "cuda", 
) -> AutoEncoder:
    """
    Load an AutoEncoder model.
    
    Args:
        name: Model name or path
        pretrained_ckpt_path: Path to checkpoint (overrides name)
        device: Device to load model on
        
    Returns:
        model: Loaded AutoEncoder model
    """
    logger.info(f"Loading AutoEncoder model: {name}")
    
    # Convert device string to torch.device if needed
    if isinstance(device, str):
        device = torch.device(device)

    # Resolve checkpoint path
    ckpt_path = resolve_model_path(
        name=name,
        repo_id_field="repo_id",
        filename_field="filename",
        ckpt_path_field="ckpt_path",
        hf_download=True,
    )

    # Initialize model
    with device:
        ae = AutoEncoder(AE_PARAMS)

    # Load weights
    model = load_model_weights(ae, ckpt_path, device=device, strict=False)

    return model


def load_redux(
    redux_name: str = "flux1-redux-dev", 
    siglip_name: str = "siglip-so400m-patch14-384",
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> ReduxImageEncoder:
    """
    Load a Redux Image Encoder model.
    
    Args:
        redux_name: Redux model name or path
        siglip_name: SigLIP model name or path
        device: Device to load model on
        dtype: Data type for model
        
    Returns:
        model: Loaded Redux Image Encoder model
    """
    logger.info(f"Loading Redux Image Encoder: redux={redux_name}, siglip={siglip_name}")
    
    # Convert device string to torch.device if needed
    if isinstance(device, str):
        device = torch.device(device)

    # Resolve Redux path
    redux_path = resolve_model_path(
        name=redux_name,
        repo_id_field="repo_id",
        filename_field="flow_filename",
        ckpt_path_field="ckpt_path",
        hf_download=True,
    )

    # Resolve SigLIP path - for SigLIP we don't need to download specific files,
    # just need the repo_id or local path
    siglip_path = resolve_model_path(
        name=siglip_name,
        repo_id_field="repo_id",
        filename_field=None,  # No specific file needed
        ckpt_path_field="ckpt_path",
        hf_download=False,  # ReduxImageEncoder handles SigLIP downloads itself
    )

    # Initialize and return the model
    with device:
        image_encoder = ReduxImageEncoder(
            redux_path=redux_path, 
            siglip_path=siglip_path,
            device=device,
        ).to(dtype=dtype)

    return image_encoder



# -------------------------------------------------------------------------
# 3) load and save weights func
# -------------------------------------------------------------------------
def save_lora_weights(model, save_path):
    """
    Extracts LoRA weights from the given model and saves them as a safetensors file.

    Args:
        model (torch.nn.Module): The model containing LoRA weights.
        save_path (str): The path to save the safetensors file.
    """

    # Collect LoRA weights (commonly containing '_lora' in their names)
    lora_state_dict = {}
    for name, param in model.state_dict().items():
        if '_lora' in name:
            lora_state_dict[name] = param.cpu()
    if not lora_state_dict:
        logger.warning("No LoRA weights found in the model to save.")
    save_sft(lora_state_dict, save_path)
    logger.info(f"LoRA weights saved to {save_path}")


def save_txt_img_in_weights(model, save_path):
    """
    Save the weights and biases of 'txt_in' and 'img_in' layers from the model.

    This function extracts parameters whose names are:
        - 'txt_in.weight'
        - 'txt_in.bias'
        - 'img_in.weight'
        - 'img_in.bias'
    and saves them to a safetensors file.

    Args:
        model (torch.nn.Module): The model containing the parameters.
        save_path (str): The file path to save the extracted weights.
    """
    target_keys = ['txt_in.weight', 'txt_in.bias', 'img_in.weight', 'img_in.bias']
    selected_state_dict = {}
    for name, param in model.state_dict().items():
        if name in target_keys:
            selected_state_dict[name] = param.cpu()
    if not selected_state_dict:
        logger.warning("No txt_in/img_in weights or biases found in the model to save.")
    save_sft(selected_state_dict, save_path)
    logger.info(f"txt_in/img_in weights and biases saved to {save_path}")


def save_task_rigister_embeddings(weights, save_path):
    """
    Save the weights and biases of 'mask_type_embedding' layer from the model.
    """
    target_keys = ['task_register_embeddings.weight']
    selected_state_dict = {}
    for name, param in weights.items():
        if name in target_keys:
            selected_state_dict[name] = param.cpu()
    if not selected_state_dict:
        logger.warning("No task_register_embeddings weights found in the model to save.")
    save_sft(selected_state_dict, save_path)
    logger.info(f"task_register_embeddings weights saved to {save_path}")


def save_boundary_embeddings(weights, save_path):
    """
    Save the weights and biases of 'boundary_embedding' layer from the model.
    """
    target_keys = ['cond_embedding.weight', 'target_embedding.weight', 'idx_embedding.weight']
    selected_state_dict = {}
    for name, param in weights.items():
        if name in target_keys:
            selected_state_dict[name] = param.cpu()
    if not selected_state_dict:
        logger.warning("No boundary_embedding weights found in the model to save.")
    save_sft(selected_state_dict, save_path)
    logger.info(f"boundary_embedding weights saved to {save_path}")


def load_model_weights(
    model, 
    weights_path, 
    device=None, 
    strict=False, 
    assign=True, 
    filter_keys=False
):
    """
    Unified function to load weights into a model from a safetensors file.
    
    Args:
        model (torch.nn.Module): The model to update with weights.
        weights_path (str): Path to the safetensors file containing weights.
        device (str or torch.device, optional): Device to load weights on. If None, uses CPU.
        strict (bool): Whether to strictly enforce that the keys match.
        assign (bool): Whether to assign weights (used by some models).
        filter_keys (bool): If True, only loads keys that exist in the model.
        
    Returns:
        model: The model with weights loaded
    """
    if weights_path is None:
        logger.info("No weights path provided, skipping weight loading")
        return model
    
    logger.info(f"Loading weights from {weights_path}")
    
    # Load the state dict
    if device is not None:
        # load_sft doesn't support torch.device objects
        device_str = str(device) if not isinstance(device, str) else device
        state_dict = load_sft(weights_path, device=device_str)
    else:
        state_dict = load_sft(weights_path)
    
    # Handle different loading strategies
    if filter_keys:
        # Filter keys to only those in the model
        model_state_dict = model.state_dict()
        update_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
        missing_keys = [k for k in state_dict if k not in model_state_dict]
        if missing_keys:
            logger.warning(f"Some keys in the file are not found in the model: {missing_keys}")
        missing, unexpected = [], []
        model.load_state_dict(update_dict, strict=strict)
    else:
        # Standard loading
        missing, unexpected = model.load_state_dict(state_dict, strict=strict, assign=assign)
    
    # Report any issues with loading
    if len(unexpected) > 0:
        print_load_warning(unexpected=unexpected)
    
    return model



def load_safetensors(path):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors
