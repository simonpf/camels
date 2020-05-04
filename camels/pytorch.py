"""
The pytorch module provides pytorch datasets providing access to the streamflow
data.
"""
import torch
from torch import nn
from torch.utils.data import Dataset
from camels.data import (StreamflowDataset,
                         DataLoader,
                         default_inputs)

class Streamflow(StreamflowDataset, Dataset):
    """
    Pytorch dataset instance providing an interface to the
    streamflow data.
    """
    def __init__(self,
                 gauge_id,
                 mode,
                 sequence_length=400,
                 stride=40,
                 inputs=default_inputs):
        StreamflowDataset.__init__(self,
                                   gauge_id,
                                   mode,
                                   sequence_length=sequence_length,
                                   stride=stride,
                                   inputs=inputs)
        Dataset.__init__(self)

        self.x = torch.zeros(self.sequence_length, len(self), len(self.inputs))
        self.y = torch.zeros(self.sequence_length, len(self), 1)
        for i in range(len(self)):
            sample = StreamflowDataset.__getitem__(self, i)
            self.x[:, i, :] = torch.tensor(sample[0]).float()
            self.y[:, i, :] = torch.tensor(sample[1]).float()

    def __getitem__(self, i):
        """
        Get sample from dataset as torch.tensors.

        A single sample corresponds to a time series of forcings and corresponding stream
        flows of length :code:`sequence_length`.

        Args:
            i (int): The index of the sample. Idexing is in temporal order with a stride
                of 10.

        Returns:
            Tuple x, y of torch tensors corresponding to the time series of forcings x
            and corresponding stream flow y.
        """
        x = self.x[:, i]
        y = self.y[:, i]
        return x, y

    def get_range(self,
                  start=None,
                  end=None):
        """
        Returns time series of given range as sample.

        Returns:
            Tuple (x, y) of torch tensors containing forcings x and streamflow y for the
            given time range.
        """
        x, y = StreamflowDataset.get_range(self, start, end)
        x = torch.tensor(x, dtype=torch.float)
        x = x.unsqueeze(dim=0)
        y = torch.tensor(y, dtype=torch.float)
        y = y.unsqueeze(dim=0)
        return x, y

    def data_loader(self,
                    batch_size,
                    shuffle=True):
        """
        Return a data loader to provide batched access to the samples in the data.

        Args:
            batch_size(int): The batch size to of the data loader.
            shuffle: Whether or not to shuffle samples in the data.
        """
        return DataLoader(self,
                          batch_size,
                          shuffle=shuffle)
