# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import pytest
from unittest.mock import patch, MagicMock

from tvln.clip_features import CLIPFeatures


@pytest.fixture
def dummy_image_file():
    return {"text": ["a"], "image": ["b"]}


@patch("tvln.clip_features.cleanup")
@patch.object(CLIPFeatures, "extract", return_value=MagicMock(name="tensor"))
@patch.object(CLIPFeatures, "set_model_link")
@patch.object(CLIPFeatures, "set_model_type")
@patch.object(CLIPFeatures, "set_precision")
@patch.object(CLIPFeatures, "set_device")
def test_clip_features_flow(mock_dev, mock_prec, mock_type, mock_link, mock_extract, mock_cleanup, dummy_image_file):
    # run the three blocks (copy‑paste the original snippet here)
    # block 1
    from tvln.main import main

    tensor_stack = main()
    # assertions
    assert mock_dev.call_count == 3
    assert mock_prec.call_count == 3
    assert mock_link.call_count == 2
    assert mock_type.call_count == 1
    assert mock_extract.call_count == 3
    assert mock_cleanup.call_count == 3
    for name, tensors in tensor_stack.items():
        if name != "F1 VAE":
            assert isinstance(tensors[1], MagicMock)  # extraction was triggered
