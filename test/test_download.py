import pytest
from camels.data import StreamflowDataset

def test_download():
    gauge_id = 13331500
    training_data = StreamflowDataset(gauge_id, "training")
    validation_data = StreamflowDataset(gauge_id, "validation")
    test_data = StreamflowDataset(gauge_id, "testing")
    training_data[0]
    validation_data[0]
    test_data[0]


