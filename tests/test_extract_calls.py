# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import pytest
from unittest.mock import MagicMock, patch
from tvln.extract import FeatureExtractor
from tvln.options import FloraModel, OpenClipModel


@pytest.fixture
def mock_flora_model():
    return MagicMock(spec=FloraModel)


@pytest.fixture
def mock_open_clip_model():
    return MagicMock(spec=OpenClipModel)


@pytest.fixture
def mock_device_type():
    return MagicMock(type=str)


@pytest.fixture
def mock_feature_extractor():
    # Use a real FeatureExtractor but replace its internal extractors with mocks
    image = MagicMock()
    extractor = FeatureExtractor(image)

    # Mock the three private extraction helpers
    extractor._extract_flora = MagicMock(name="_extract_flora")
    extractor._extract_openclip = MagicMock(name="_extract_openclip")
    extractor._extract_vae = MagicMock(name="_extract_vae")
    return extractor


def test_extract_with_flora_model(mock_feature_extractor, mock_flora_model, mock_device_type):
    import torch

    mock_device_type.type = "cpu"
    mock_flora_model.value = ("mock_flora_hub_url", "mock_flora_url")
    mock_feature_extractor.model = mock_flora_model
    mock_feature_extractor.encoder = MagicMock()
    mock_feature_extractor.image.tensor = MagicMock(dtype=torch.float32, device=mock_device_type)

    # Make the mocked _extract_flora return a dummy tensor
    dummy_tensor = MagicMock()
    mock_feature_extractor._extract_flora.return_value = dummy_tensor

    tensor, _ = mock_feature_extractor.extract(mock_flora_model)

    # Verify the helper was called with the correct device/precision setup
    mock_feature_extractor._extract_flora.assert_called_once()
    assert tensor is dummy_tensor


def test_extract_with_open_clip_model(mock_feature_extractor, mock_open_clip_model, mock_device_type):
    import torch

    mock_device_type.type = "cpu"
    mock_open_clip_model.value = ("EVA02-B-16", "MERGED2B_S8B_B131K")
    mock_feature_extractor.model = mock_open_clip_model
    mock_feature_extractor.encoder = MagicMock()
    mock_feature_extractor.image.tensor = MagicMock(dtype=torch.float32, device=mock_device_type)

    dummy_tensor = MagicMock()
    mock_feature_extractor._extract_openclip.return_value = dummy_tensor

    tensor, _ = mock_feature_extractor.extract(mock_open_clip_model)

    mock_feature_extractor._extract_openclip.assert_called_once()
    assert tensor is dummy_tensor


def test_cleanup_method(mock_feature_extractor):
    # No change needed; this test already mocks GPU cleanup
    mock_feature_extractor.encoder = MagicMock()
    mock_feature_extractor.image.tensor.device.type = "cpu"

    with patch("torch.cuda.empty_cache") as mock_empty_cache, patch("gc.collect") as mock_gc_collect:
        mock_feature_extractor.cleanup()
        mock_empty_cache.assert_not_called()
        mock_gc_collect.assert_called_once()
