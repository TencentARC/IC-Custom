import cv2
import numpy as np
import torch
from typing import Union
from einops import rearrange, repeat
from PIL import Image
from safetensors.torch import load_file as load_sft
from torch import nn
from transformers import AutoModelForDepthEstimation, AutoProcessor, SiglipImageProcessor, SiglipVisionModel

from ..utils.process_util import print_load_warning

class ReduxImageEncoder(nn.Module):

    def __init__(
        self,
        redux_path: str,
        siglip_path: str = "google/siglip-so400m-patch14-384",
        redux_dim: int = 1152,
        txt_in_features: int = 4096,
        device: Union[str, torch.device, None] = None,
    ) -> None:
        super().__init__()

        self.redux_dim = redux_dim

        self.redux_up = nn.Linear(redux_dim, txt_in_features * 3)
        self.redux_down = nn.Linear(txt_in_features * 3, txt_in_features)

        sd = load_sft(redux_path)
        missing, unexpected = self.load_state_dict(sd, strict=False, assign=True)
        print_load_warning(missing, unexpected)

        self.siglip = SiglipVisionModel.from_pretrained(siglip_path)
        self.normalize = SiglipImageProcessor.from_pretrained(siglip_path)

        self.to(device)

    def __call__(self, x: Image.Image, device: Union[str, torch.device, None] = None, dtype: Union[str, torch.dtype, None] = None) -> torch.Tensor:
        if isinstance(device, str):
            device = torch.device(device)

        if isinstance(dtype, str):
            dtype = torch.dtype(dtype)

        if device is None:
            device = next(self.parameters()).device

        if dtype is None:
            dtype = next(self.parameters()).dtype

        imgs = self.normalize.preprocess(images=[x], do_resize=True, return_tensors="pt", do_convert_rgb=True)
        _encoded_x = self.siglip(**imgs.to(device=device, dtype=dtype)).last_hidden_state
        projected_x = self.redux_down(nn.functional.silu(self.redux_up(_encoded_x)))

        return projected_x