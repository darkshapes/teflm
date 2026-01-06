# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


from torch import Tensor, nn, dtype, float32, device

from tvln.batch import ImageFile
from tvln.options import DeviceName, FloraModel, OpenClipModel


class FloraEncoder(nn.Module):
    """CLIP wrapper\n\n
    MIT licensed by ncclab-sustech/BrainFLORA
    """

    def __init__(self, device: str = DeviceName.CPU) -> None:
        """Instantiate the encoder with a specific device and model\n
        :param device: The graphics device to allocate, Default is cpu"""
        super().__init__()
        self.flora_model, _ = FloraModel.VIT_L_14_LAION2B_S32B_B82K.value  # type: ignore dynamic
        self.device = device

    def clip_encode_image(self, x: Tensor) -> Tensor:
        """Encode image patches using CLIP vision model\n
        Include class and positional embedding, then stop at second-to-last layer where features are strongest\n
        :param x: Tensors of the image being processed"""

        import torch

        x = x.reshape(x.shape[0], x.shape[1], -1)  # [batchsize, 1024, 256]
        x = x.permute(0, 2, 1)

        class_embedding = self.model.vision_model.embeddings.class_embedding.to(x.dtype)
        class_embedding = class_embedding.repeat(x.shape[0], 1, 1)  # [batchsize, 1, 1024]
        x = torch.cat([class_embedding, x], dim=1)

        pos_embedding = self.model.vision_model.embeddings.position_embedding
        position_ids = torch.arange(0, 257).unsqueeze(0).to(self.device)
        x = x + pos_embedding(position_ids)

        x = self.model.vision_model.pre_layrnorm(x)
        x = self.model.vision_model.encoder(x, output_hidden_states=True)

        select_hidden_state_layer = -2
        select_hidden_state = x.hidden_states[select_hidden_state_layer]  # [1, 256, 1024] #type:ignore came with code
        image_features = select_hidden_state[:, 1:]  # Remove class token

        return image_features

    def encode_image(self, x: Tensor) -> Tensor:
        """Full image encoding pipeline
        :param x: the input image tensor in shape [B, C, H, W] and device-compatible dtype."""
        from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize
        from transformers import CLIPVisionModel

        self.model = CLIPVisionModel.from_pretrained(pretrained_model_name_or_path=self.flora_model).to(self.device)  # type: ignore DeviceLikeType

        self.clip_size = (224, 224)
        self.preprocess = Compose(
            [
                Resize(size=self.clip_size[0], interpolation=InterpolationMode.BICUBIC),
                CenterCrop(size=self.clip_size),
                Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ]
        )
        x = x.to(self.device)
        x = self.preprocess(x)  # [3, 224, 224]
        x = self.model.vision_model.embeddings.patch_embedding(x)  # [1024, 16, 16]
        image_feats = self.clip_encode_image(x)
        return image_feats


class OpenClipEncoder:
    FLOAT64 = "fp64"
    FLOAT32 = "fp32"
    BFLOAT16 = "bf16"
    FLOAT16 = "fp16"

    def __init__(self, device: str | device = DeviceName.CPU, precision: dtype = float32) -> None:
        super().__init__()
        self.open_clip_model, self.pretraining = OpenClipModel.VIT_L_14_LAION2B_S32B_B82K.value  # type:ignore
        self.precision: dtype = precision
        self.device = device

    def convert_dtype(self, precision: dtype) -> str:
        import torch

        if isinstance(precision, torch.dtype):
            _, torch_dtype = precision.__repr__().rsplit(".")
            torch_dtype = getattr(self, torch_dtype.upper())
        return torch_dtype

    def encode_image(self, x: ImageFile) -> Tensor:
        """Encode a batch of images into CLIP features.\n
        :param images: Paths to the image files.
        :returns Concatenated image feature vectors."""

        from open_clip import create_model_and_transforms
        from PIL import Image
        from torch import cat as torch_cat
        from torch import no_grad as torch_no_grad
        from torch import stack as torch_stack

        self.images = [x.image_path]
        vlmodel, preprocess_train, feature_extractor = create_model_and_transforms(
            self.open_clip_model,
            pretrained=self.pretraining,
            precision=self.convert_dtype(self.precision),
            device=self.device,
        )

        batch_size = 512
        image_features_list = []

        for i in range(0, len(self.images), batch_size):
            batch_images = self.images[i : i + batch_size]
            image_inputs = torch_stack([preprocess_train(Image.open(img).convert("RGB")) for img in batch_images])  # type:ignore

            with torch_no_grad():
                batch_image_features = vlmodel.encode_image(image_inputs)  # type: ignore came with code
            image_features_list.append(batch_image_features)

        image_features = torch_cat(image_features_list, dim=0)
        return image_features
