# import pytest
# from PIL import Image

# from tvln.clip_features import DeviceName, ImageFile
# from tvln.options import ModelType


# def test_clip_features_init():
#     clip = CLIPFeatures()
#     assert clip._model_type == "ViT-L-14"
#     assert clip._pretrained == "laion2b_s32b_b82k"
#     assert clip._precision == "fp32"


# def test_set_device():
#     clip = CLIPFeatures()
#     clip.set_device(DeviceName.CUDA)
#     assert clip._device == DeviceName.CUDA.value


# def test_extract(tmp_path):
#     import torch

#     img_file = tmp_path / "test.jpg"
#     Image.new("RGB", (224, 224)).save(img_file)
#     image = ImageFile()
#     image._default_path = img_file
#     clip = CLIPFeatures()
#     features = clip.extract(image)
#     assert isinstance(features, torch.Tensor)
#     assert features.shape == (512,)  # Example shape based on model output


# @pytest.mark.parametrize("precision", ["fp16", "fp32"])
# def test_set_precision(precision):
#     clip = CLIPFeatures()
#     clip.set_precision(PrecisionType[precision])
#     assert clip._precision == precision
