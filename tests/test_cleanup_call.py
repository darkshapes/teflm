import pytest
from unittest.mock import patch
from tvln.extract import FeatureExtractor, ImageFile
from tvln.clip_features import FloraEncoder
import torch


@pytest.fixture
def dupe_image():
    image = ImageFile()
    image.tensor = torch.randn((1, 3, 256, 256))
    return image


class TestFeatureExtractor:
    @patch("tvln.extract.snapshot_download")
    def test_cleanup(self, mock_download, dupe_image):
        extractor = FeatureExtractor(dupe_image)
        extractor.encoder = FloraEncoder()
        with patch("torch.cuda.empty_cache", return_value=None) as mock_empty_cache, patch("gc.collect") as mock_collect:
            extractor.cleanup()
            if dupe_image.tensor.device.type != "cpu":
                mock_empty_cache.assert_called_once()
            mock_collect.assert_called_once()
