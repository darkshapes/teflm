# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import torch


@torch.no_grad
def main():
    from tvln.batch import ImageFile
    from tvln.options import DeviceName, OpenClipModel, FloraModel
    from tvln.extract import FeatureExtractor

    device = DeviceName.CPU
    if torch.cuda.is_available():
        device = DeviceName.CUDA
    elif torch.mps.is_available():
        device = DeviceName.MPS

    image_file = ImageFile()
    image_file.single_image()
    image_file.as_tensor(device=DeviceName.CPU, dtype=torch.float32)

    feature_extractor = FeatureExtractor(image=image_file)
    clip_l_tensor, clip_l_data = feature_extractor.extract(model=FloraModel.VIT_L_14_LAION2B_S32B_B82K)  # type:ignore cannot access
    clip_l_e32_tensor, clip_l_e32_data = feature_extractor.extract(model=OpenClipModel.VIT_L_14_LAION400M_E32)  # type:ignore cannot access
    clip_g_tensor, clip_g_data = feature_extractor.extract(model=FloraModel.VIT_BIGG_14_LAION2B_S39B_B160K)  # type:ignore cannot access
    image_file.as_tensor(device=device, dtype=torch.float32)
    vae_tensor, vae_data = feature_extractor.extract(model="black-forest-labs/FLUX.1-dev")

    aggregate_data = {
        "CLIP_L_S32B82K": [clip_l_data, clip_l_tensor],
        "CLIP_L_400M_E32": [clip_l_e32_data, clip_l_e32_tensor],
        "CLIP_BIG_G_S39B_B160K": [clip_g_data, clip_g_tensor],
        "F1 VAE": [vae_data, vae_tensor],
    }
    print(aggregate_data)
    return aggregate_data


if __name__ == "__main__":
    main()
