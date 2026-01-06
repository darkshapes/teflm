# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


from enum import Enum

from open_clip import list_pretrained
from open_clip.pretrained import _PRETRAINED


class DeviceName(str, Enum):
    """Graphics processors usable by the CLIP pipeline."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class OpenClipModels:
    """An OpenCLIP model configuration
    Tuple of (model_type, pretrained)
    :model_type: str: Name of the model
    :pretrained: str: Name of the pretraining dataset"""

    pass


class FloraModels:
    """A pretrained model for image feature extraction
    hf_hub: Huggingface Hub URL to the model
    url: Alternate url to the model
    Tuple of (hf_hub:str, url:str)"""

    pass


FloraModel = Enum(
    "FloraModel",
    {
        f"{family.replace('-', '_').upper()}_{id.replace('-', '_').upper()}": (data.get("hf_hub", "").strip("/"), data.get("url"))
        for family, name in _PRETRAINED.items()
        for id, data in name.items()
        if data.get("hf_hub") or data.get("url")
    },
    module=type(OpenClipModels),
)


OpenClipModel = Enum(
    "OpenClipModel",
    {
        # member name → (model_type, pretrained) value
        f"{model.replace('-', '_').upper()}_{pretrained.replace('-', '_').upper()}": (
            model,
            pretrained,
        )
        for model, pretrained in list_pretrained()
    },
    module=type(FloraModels),
)
