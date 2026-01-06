# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from enum import Enum

import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from huggingface_hub import snapshot_download

from tvln.batch import ImageFile
from tvln.clip_features import FloraEncoder, OpenClipEncoder
from tvln.options import FloraModel, OpenClipModel


class FeatureExtractor:
    encoder: FloraEncoder | OpenClipEncoder | AutoencoderKL = OpenClipEncoder()

    def __init__(self, image: ImageFile):
        self.image: ImageFile = image

    def extract(self, model: Enum | str) -> tuple[torch.Tensor, str | dict]:
        """Extract features from the image using the specified model.
        :param model_info: The kind of model to use
        :param image: One or more image file paths.
        :returns: Extracted image features"""

        self.dtype = self.image.tensor.dtype
        self.device = self.image.tensor.device
        self.model = model

        if isinstance(model, FloraModel):  # type: ignore
            tensor = self._extract_flora()
        elif isinstance(model, OpenClipModel):  # type: ignore
            tensor = self._extract_openclip()
        else:
            tensor = self._extract_vae()

        data: dict = {"model": model, "dtype": self.image.tensor.dtype, "device": self.image.tensor.device} | vars(self.image)
        self.cleanup()
        return tensor, data

    def _extract_flora(self) -> torch.Tensor:
        """Extract features using a Flora model."""
        flora_encoder = FloraEncoder(device=self.device.type)
        flora_encoder.flora_model, _ = self.model.value  # type: ignore
        tensor: torch.Tensor = flora_encoder.encode_image(self.image.tensor)
        self.encoder = flora_encoder
        return tensor

    def _extract_openclip(self) -> torch.Tensor:
        """Extract features using an OpenClip model."""
        open_clip_encoder = OpenClipEncoder(device=self.device.type, precision=self.dtype)
        open_clip_encoder.open_clip_model, open_clip_encoder.pretraining = self.model.value  # type: ignore
        open_clip_encoder.precision = self.image.tensor.dtype
        tensor: torch.Tensor = open_clip_encoder.encode_image(self.image)
        self.encoder = open_clip_encoder
        return tensor

    def _extract_vae(self) -> torch.Tensor:
        """Extract features using a VAE model."""
        import os

        vae_path = snapshot_download(self.model, allow_patterns=["vae/*"])  # type:ignore
        vae_path = os.path.join(vae_path, "vae")
        vae_model = AutoencoderKL.from_pretrained(vae_path, torch_dtype=self.dtype).to(self.device.type)  # type:ignore DeviceLike
        vae_tensor = vae_model.tiled_encode(self.image.tensor, return_dict=False)
        tensor = vae_tensor[0].sample()
        self.encoder = vae_model
        return tensor

    def cleanup(self) -> None:  # type:ignore
        """Cleans up the model and frees GPU memory
        :param model: The model instance used for feature extraction"""

        import gc

        device = self.image.tensor.device.type
        if device != "cpu":
            gpu = getattr(torch, device)
            gpu.empty_cache()
        del self.encoder
        gc.collect()
