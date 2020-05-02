import os
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm

def read_forcing_file(filename):
    """
    Reads a mean basin forcing file into a pandas Dataframe.

    Args:
        filename: Filename of the text file from the original data
            to read.

    Returns:
        Pandas dataframe containing the file contents.
    """
    names = ["year",
             "month",
             "day",
             "hour",
             "day length [s]",
             "precipitation [mm/d]",
             "solar radiation [W/m^2]",
             "snow water equivalent [mm]",
             "maximum temperature [C]",
             "minimum temperature [C]",
             "vapour pressure [Pa]"]
    forcing = pd.read_csv(filename,
                          header=3,
                          index_col="time",
                          names=names,
                          parse_dates={"time" : [0, 1, 2]},
                          delim_whitespace=True)
    forcing = forcing.drop(columns=["hour"])
    return forcing

def read_streamflow_file(filename):
    """
    Reads a streamflow file into a pandas Dataframe.

    Args:
        filename: Filename of the text file from the original data
            to read.

    Returns:
        Pandas dataframe containing the streamflow data.
    """
    names=["gauge id", "year", "month", "day", "streamflow [ft^3/s]", "quality flag"]
    streamflow = pd.read_csv(filename,
                         parse_dates={"time" : [1, 2, 3]},
                         names=names,
                         delim_whitespace=True,
                         index_col="time")
    return streamflow

def read_gauge_information(filename):
    """
    Reads metadata file with gauge information.

    Args:
        filename: Filename of the text file from the original data
            to read.
    Returns:
        Pandas dataframe containing gauge meta data.
    """
    names = ["hydrologic unit code",
             "gauge id",
             "gauge name",
             "latitude",
             "longitude",
             "drainage area"]
    gauge_information = pd.read_csv(filename,
                                    names=names,
                                    skiprows=1,
                                    sep=";")
    return gauge_information

def convert_to_binary(base_path,
                      output_path):
    """
    Converts files from original dataset to pandas dataframes and stores
    them using pickle.

    Arguments:
        base_path: The path containing the original dataset
        output_path: The path to which to store the converted data.
    """

    base_path = os.path.expanduser(base_path)
    output_path = os.path.expanduser(output_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Read gauge information
    path = os.path.join(base_path, "basin_metadata", "gauge_information.txt")
    gauge_information = read_gauge_information(path)
    gauge_information.to_pickle(os.path.join(output_path, "gauge_information.pckl"))

    # Forcing files
    forcing_folder = os.path.join(output_path, "forcing")
    if not os.path.exists(forcing_folder):
        os.makedirs(forcing_folder)
    files = glob.glob(os.path.join(base_path, "basin_mean_forcing", "daymet", "*", "*_forcing_leap.txt"))
    print("Converting forcing files:")
    for f in tqdm(files):
        name = os.path.basename(f)
        name = name.split("_")[0]
        data = read_forcing_file(f)
        data.to_pickle(os.path.join(forcing_folder, name + ".pckl"))

    # Streamflow files
    streamflow_folder = os.path.join(output_path, "streamflow")
    if not os.path.exists(streamflow_folder):
        os.makedirs(streamflow_folder)
    files = glob.glob(os.path.join(base_path, "usgs_streamflow", "*", "*_streamflow_qc.txt"))
    print("Converting streamflow files:")
    for f in tqdm(files):
        name = os.path.basename(f)
        name = name.split("_")[0]
        data = read_streamflow_file(f)
        data.to_pickle(os.path.join(streamflow_folder, name + ".pckl"))

#convert_to_binary("~/Downloads/basin_dataset_public_v1p2/", "~/src/camels/data")
