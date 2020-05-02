"""
The data module contains functions to download to the local machine and a
baseclass for datasets of streamflow data.
"""
from appdirs import AppDirs
from urllib import request
from datetime import datetime
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

################################################################################
# Retrieving files
################################################################################

_app_dirs = AppDirs("camels", "simonpf")
_base_dir = _app_dirs.user_data_dir
_forcing_dir = os.path.join(_base_dir, "forcing")
_streamflow_dir = os.path.join(_base_dir, "streamflow")

for d in [_base_dir, _forcing_dir, _streamflow_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

gi_file = os.path.join(_base_dir, "gauge_information.pckl")
if not os.path.exists(os.path.join(_base_dir, "gauge_information.pckl")):
    url = "http://spfrnd.de/datasets/camels/gauge_information.pckl"
    with request.urlopen(url) as response:
        shutil.copyfileobj(response, open(gi_file, "wb"))
gauge_information = pd.read_pickle(gi_file)
gauge_ids = gauge_information["gauge id"].array

def get_forcing_file(gauge_id):
    """
    Get forcing file for given gauge ID. If the file is not found in the local
    cache it is downloaded from spfrnd.de/datasets/camels.

    The provided forcing data is the basin-mean forcing derived from Daymet data.

    Args:
        gauge_id: String or integer representing the gauge for which
        to retrieval the forcing data.

    Returns:
        pandas Dataframe containing the full forcing time series.
    """
    if not gauge_id in gauge_ids:
        other = gauge_ids[np.argmin(np.abs(int(gauge_id) - gauge_ids))]
        raise ValueError("Gauge ID {} is not available from this dataset. The "
                         "closest ID available is {}.".format(gauge_id, other))

    forcing_file = os.path.join(_forcing_dir, "{}.pckl".format(gauge_id))
    if not os.path.exists(forcing_file):
        url = "http://spfrnd.de/datasets/camels/forcing/{:08d}.pckl".format(gauge_id)
        with request.urlopen(url) as response:
            shutil.copyfileobj(response, open(forcing_file, "wb"))
    return pd.read_pickle(forcing_file)

def get_streamflow_file(gauge_id):
    """
    Get streamflow file for given gauge ID. If the file is not found in the local
    cache it is downloaded from spfrnd.de/datasets/camels.

    Args:
        gauge_id: String or integer representing the gauge for which
        to retrieval the forcing data.

    Returns:
        pandas Dataframe containing the full streamflow time series.
    """
    if not gauge_id in gauge_ids:
        other = gauge_ids[np.argmin(np.abs(int(gauge_id) - gauge_ids))]
        raise ValueError("Gauge ID {} is not available from this dataset. The "
                         " closest ID available is {}.".format(gauge_id, other))

    streamflow_file = os.path.join(_streamflow_dir, "{}.pckl".format(gauge_id))
    if not os.path.exists(streamflow_file):
        url = "http://spfrnd.de/datasets/camels/streamflow/{:08d}.pckl".format(gauge_id)
        with request.urlopen(url) as response:
            shutil.copyfileobj(response, open(streamflow_file, "wb"))
    return pd.read_pickle(streamflow_file)

################################################################################
# Retrieving files
################################################################################

default_inputs = ["day length [s]",
                  "precipitation [mm/d]",
                  "solar radiation [W/m^2]",
                  "maximum temperature [C]",
                  "minimum temperature [C]",
                  "vapour pressure [Pa]"]

all_inputs = ["day length [s]",
              "precipitation [mm/d]",
              "solar radiation [W/m^2]",
              "snow water equivalent [mm]",
              "maximum temperature [C]",
              "minimum temperature [C]",
              "vapour pressure [Pa]"]

class StreamflowDataset:
    """
    Base class for streamflow datasets.

    Attributes:
        gauge_id (int): The gauge ID of the gauge the data is taken from.
        mode (str): "training" or "testing" depending on whether the dataset contains
            training data (<= 2010-01-01) or test data (> 2010-01-01)
        inputs: Column names of the forcing dataframe used as input data.
        outputs: Name of the output column (streamflow [ft^3/s])
        means: The means of all inputs and outputs
        stds: The standard deviations of all inputs and outputs
        data: Dataframe containing the input and output data combined.
        sequence_length: The length of the sequences to provide as training data.
    """
    def __init__(self,
                 gauge_id,
                 mode,
                 sequence_length=300,
                 inputs=default_inputs):
        """

        """

        if not gauge_id in gauge_ids:
            other = gauge_ids[np.argmin(np.abs(int(gauge_id) - gauge_ids))]
            raise ValueError("Gauge ID {} is not available from this dataset. The "
                            "closest ID available is {}.".format(gauge_id, other))

        self.gauge_id = gauge_id
        self.mode = mode
        self.inputs = inputs
        outputs = ["streamflow [ft^3/s]"]
        self.outputs = outputs

        forcing = get_forcing_file(gauge_id)
        streamflow = get_streamflow_file(gauge_id)

        data = pd.DataFrame()
        data[inputs] = forcing[inputs]
        data[outputs] = streamflow[outputs]

        data = data.loc[data[outputs[0]] >= 0.0]
        self.means = data.mean()
        self.stds = data.std()

        if mode == "training":
            self.data = data.loc[data.index < datetime(2006, 1, 1, hour=0)]
        elif mode == "validation":
            self.data = data.loc[(data.index >= datetime(2006, 1, 1, hour=0))
                                 & (data.index < datetime(2010, 1, 1, hour=0))]
        elif mode == "testing":
            self.data = data.loc[data.index >= datetime(2010, 1, 1, hour=0)]
        else:
            raise ValueError("The 'mode' argument should be one of 'training', "
                             "'validation' or 'testing'.")

        self.sequence_length = sequence_length

    def __repr__(self):
        return "Streamflow {} dataset for gauge {}".format(self.mode, self.gauge_id)

    def __len__(self):
        """
        The number of samples in the dataset.

        The length of the dataset is defined as the number of days with valid data
        minus the sequence length  and divided by 10. This is because an element in
        the dataset is a sequence of length :code:`sequence_length`. The element access
        uses a stride of 10 to reduce overlap between consecutive sequences.

        Returns:
            Number of samples in the dataset.
        """
        return (len(self.data) - self.sequence_length) // 10

    def __getitem__(self, i):
        """
        Return sequence of forcings and corresponding streamflow.

        A sample in the dataset is defined a sequence of length :code:`sequence_length` of
        forcings and corresponding streamflow. Elements are accessed in temporal order
        and a strid of 10. That means dataset[0] will return a sequence starting a the first
        day of the data, while dataset[1] will return a sequence starting a the 10th day
        of the available data.

        Return:
            Tuple (x, y) of numpy arrays containing the forcings x and streamflow y.
        """
        if i >= len(self):
            return ValueError()

        i = i * 10
        i_start = i
        i_end = i + self.sequence_length
        inputs = ["daylight", "precipitation", "solar_radiation", "t_max", "t_min", "vapor_pressure"]
        outputs = ["streamflow [ft^3/s]"]
        x = ((self.data[i_start : i_end] - self.means) / self.stds)[inputs].to_numpy()
        y = ((self.data[i_start : i_end] - self.means) / self.stds)[outputs].to_numpy()

        return x, y

    def get_range(self, start, end):
        """
        Returns time series of given range as sample.

        Returns:
            Tuple (x, y) of numpy arrays containing forcings x and streamflow y for the
            given time range.
        """
        data = self.data[start : end]
        inputs = ["daylight", "precipitation", "solar_radiation", "t_max", "t_min", "vapor_pressure"]
        outputs = ["streamflow"]
        x = ((data - self.means) / self.stds)[inputs].to_numpy()
        y = ((data - self.means) / self.stds)[outputs].to_numpy()

        return x, y

    def plot_overview(self, start, end):
        """
        Generates an overview plot of forcings and corresponding streamflow for
        a given time range.

        Arguments:
            start (datetime.datetime): Datetime object representing the start
                date of the range to display.
            end (datetime.datetime): Datetime object representing the end date
                of the range to plot.
        """
        data = self.data[start : end]

        f, axs = plt.subplots(4, 1, figsize=(12, 12))
        c1 = "navy"
        c2 = "firebrick"

        #
        # Daylight and solar radiation
        #

        ax = axs[0]
        if "day length [s]" in self.inputs:
            ax.plot(data["day length [s]"], c=c1)
            ax.set_ylabel("Daylight [s]")
            for l in ax.yaxis.get_ticklabels():
                l.set_color(c1)
            ax.set_xlim([data.index[0], data.index[-1]])
            ax.set_title("(a) Forcing: Incoming solar radiation", loc="left")
            ax = ax.twinx()

        if "solar radiation [W/m^2]" in self.inputs:
            ax.plot(data["solar radiation [W/m^2]"], c=c2)
            ax.set_ylabel("Solar radiance [$W\ m^{-2}$]")
            ax.set_xticks([])
            for l in ax.yaxis.get_ticklabels():
                l.set_color(c2)

        #
        # Precipitation
        #

        ax = axs[1]
        if "precipitation [mm/d]" in self.inputs:
            ax.plot(data["precipitation [mm/d]"], c=c1)
            ax.set_ylabel("Precipitation [mm]")
            ax.set_xlim([data.index[0], data.index[-1]])
            for l in ax.yaxis.get_ticklabels():
                l.set_color(c1)
            ax.set_title("(b) Forcing: Precipitation and water vapor", loc="left")
            ax = ax.twinx()

        if "vapour pressure [Pa]" in self.inputs:
            ax.plot(data["vapour pressure [Pa]"], c=c2)
            ax.set_ylabel("Vapor pressure [hPa]")
            ax.set_xticks([])
            for l in ax.yaxis.get_ticklabels():
                l.set_color(c2)

        #
        # Temperature
        #

        ax = axs[2]
        if "maximum temperature [C]" in self.inputs:
            ax.plot(data["maximum temperature [C]"], c=c2, label="Maximum")

        if "minimum temperature [C]" in self.inputs:
            ax.plot(data["minimum temperature [C]"], c=c1, label="Minimum")

        ax.set_ylabel("Temperature [C]")
        ax.set_title("(c) Forcing: Temperature", loc="left")
        ax.set_xticks([])
        ax.set_xlim([data.index[0], data.index[-1]])
        ax.legend(loc="center left", bbox_to_anchor=[1.0, 0.5])

        #
        # Streamflow
        #

        ax = axs[3]
        ax.plot(data["streamflow [ft^3/s]"], c=c1)
        ax.set_ylabel("Streamflow [$ft^3\ s^{-1}$]")
        ax.set_title("(d) Output: Streamflow", loc="left")
        ax.set_xlim([data.index[0], data.index[-1]])

        for l in ax.xaxis.get_ticklabels():
            l.set_rotation(45)
        plt.tight_layout()
        plt.show()

class DataIterator:
    """
    General data iterator providing access to sample batches.
    """
    def __init__(self,
                 data,
                 indices,
                 batch_size):
        """
        Args:
            data: DataLoader object containing the data to provide in x and
                y attributes.
            indices: The indices of the samples to load into batches.
            batch_size: The size of the batches to provide.
        """
        self.batch_size = batch_size
        self.indices = indices

    def next(self):
        if self.indices.size > self.batch_size:
            indices = self.indices[:self.batch_size]
            self.indices = self.indices[self.batch_size:]
            x = data.x[indices]
            y = data.y[indices]
            return x, y
        else:
            raise StopIteration

class DataLoader:
    """
    General DataLoader object allowing to iterate over 

    """
    def __init__(self,
                 data,
                 batch_size=8,
                 shuffle=True):
        self.batch_size = batch_size
        self.shuffle = True
        self.indices = np.arange(len(data))
        self.data = data

    def __iter__(self):
        if self.shuffle:
            indices = np.random.permutation(self.indices)
        else:
            indices = self.indices
        return DataIterator(self.data,
                            indices,
                            self.batch_size)

