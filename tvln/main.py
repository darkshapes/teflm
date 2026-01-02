# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import torch


@torch.no_grad
def main():
    import os
    from tvln.clip_features import CLIPFeatures, DeviceName, PrecisionType, ModelLink, ModelType, cleanup
    from tvln.batch import ImageFile
    from huggingface_hub import snapshot_download
    from diffusers import AutoencoderKL

    text_device: str = DeviceName.CPU
    precision = PrecisionType.FP32
    dtype = torch.float32
    if torch.cuda.is_available():
        device = DeviceName.CUDA
        precision = PrecisionType.FP32
    elif torch.mps.is_available():
        device = DeviceName.MPS
        precision = PrecisionType.FP32

    image_file = ImageFile()
    image_file.single_image()
    image_file.as_tensor(dtype=torch.float32, device=text_device)

    feature_extractor = CLIPFeatures()
    feature_extractor.set_device(text_device)
    feature_extractor.set_precision(precision)
    feature_extractor.set_model_link(ModelLink.VIT_L_14_LAION2B_S32B_B82K)  # type: ignore
    clip_l_tensor = feature_extractor.extract(image_file)
    clip_l_data = vars(feature_extractor)
    cleanup(model=feature_extractor, device=text_device)

    feature_extractor = CLIPFeatures()
    feature_extractor.set_device(text_device)
    feature_extractor.set_precision(precision)
    feature_extractor.set_model_type(ModelType.VIT_L_14_LAION400M_E32)
    clip_l_e32_tensor = feature_extractor.extract(image_file, last_layer=True)
    clip_l_e32_data = vars(feature_extractor)
    cleanup(model=feature_extractor, device=text_device)

    feature_extractor = CLIPFeatures()
    feature_extractor.set_device(text_device)
    feature_extractor.set_precision(precision)
    feature_extractor.set_model_link(ModelLink.VIT_BIGG_14_LAION2B_S39B_B160K)
    clip_g_tensor = feature_extractor.extract(image_file)
    clip_g_data = vars(feature_extractor)
    cleanup(model=feature_extractor, device=text_device)

    vae_path = snapshot_download("black-forest-labs/FLUX.1-dev", allow_patterns=["vae/*"])
    vae_path = os.path.join(vae_path, "vae")

    image_file.as_tensor(dtype=dtype, device=device)
    vae_model = AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype).to(device)
    vae_tensor = vae_model.tiled_encode(image_file.tensor, return_dict=False)

    aggregate_data = {
        "CLIP_L_S32B82K": [clip_l_data, clip_l_tensor],
        "CLIP_L_400M_E32": [clip_l_e32_data, clip_l_e32_tensor],
        "CLIP_BIG_G_S39B_B160K": [clip_g_data, clip_g_tensor],
        "F1 VAE": ["black-forest-labs/FLUX.1-dev", vae_tensor[0].sample()],
    }
    print(aggregate_data)
    return aggregate_data


if __name__ == "__main__":
    main()
