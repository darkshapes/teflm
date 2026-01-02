# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from enum import Enum
from tvln.batch import ImageFile
from tvln.clip_features import CLIPFeatures, ModelType, ModelLink, PrecisionType, DeviceName


class FeatureExtractor:
    def __init__(self, text_device: DeviceName, precision: PrecisionType):
        self.text_device = text_device
        self.precision = precision

    def extract_features(self, model_info: Enum, image_file: ImageFile, last_layer: bool = False):
        feature_extractor = CLIPFeatures()
        feature_extractor.set_device(self.text_device)
        feature_extractor.set_precision(self.precision)
        if isinstance(model_info, ModelLink):
            feature_extractor.set_model_link(model_info)
        elif isinstance(model_info, ModelType):
            feature_extractor.set_model_type(model_info)
        tensor = feature_extractor.extract(image_file, last_layer)
        data = vars(feature_extractor)
        self.cleanup(model=feature_extractor, device=self.text_device)
        return tensor, data
