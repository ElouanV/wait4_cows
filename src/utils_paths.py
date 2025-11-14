#!/usr/bin/python
# -*- coding: utf-8 -*-

# creator : Joseph Allyndree
# date : 2025-07-10

import gc
import os
from pprint import pprint as pprint

import pandas as pd
from tqdm import tqdm

###############################################################################
#                           GENERAL FUNCTIONS                                 #
###############################################################################


def filter_out_system_files(files_list):
    """
    Check if the files are system files and remove them from the list
    The following files will also be excluded :
        - files beginning with "."
        - files beginning with "$"
        - files beginning with "._"
        - files named "System Volume Information"
    Args:
        files_list (list[str]): the list of files to filter

    Returns:
        list of str: the resulting list of strings
    """
    indexes_to_pop = []
    for file in files_list:
        if file.split(f"{os.sep}")[-1].startswith("."):
            indexes_to_pop.append(files_list.index(file))
        elif file.split(f"{os.sep}")[-1].startswith("._"):
            indexes_to_pop.append(files_list.index(file))
        elif file.split(f"{os.sep}")[-1].startswith("$"):
            indexes_to_pop.append(files_list.index(file))
        elif "System Volume Information" in file.split(f"{os.sep}"):
            indexes_to_pop.append(files_list.index(file))

    # remove the files in reverse order to not mess up the indexes
    for index in sorted(indexes_to_pop, reverse=True):
        del files_list[index]

    return files_list


def get_accelero_id_from_parquet_or_csv_files(
    path_parquet_or_csv: str, only_id: str = False
) -> str:
    """
    Retrieve the id of the accelerometer from the path.

    Args:
        path_parquet_or_csv (str): the path to the accelerometer data
        only_id (bool, optional): if True, only the id is returned. Defaults to False.
    Returns:
        str: the id of the accelerometer
    """

    if path_parquet_or_csv.endswith(".parquet") or path_parquet_or_csv.endswith(".csv"):
        nom_parquet = path_parquet_or_csv.split(os.sep)[-1]
        id_accelero_long = nom_parquet.split("_")[1]
        if only_id:
            id_accelero = id_accelero_long[-4:]
            return id_accelero
        else:
            return id_accelero_long
    else:
        raise ValueError(f"The file {path_parquet_or_csv} is not a parquet nor a csv")


def get_all_files_within_one_folder(
    folder_to_search: str, ordered_by_id: bool, extension: str = "all"
):
    """
    This function is used to retrieve all the files from the directories.

    Args:
        folder_to_search (str) : the folder to search for the files
        ordered_by_id (bool) :  if True, the files will be ordered by the full id of the accelerometer (e.g. d0003cec)
                                if False, the files will be ordered by the name of the file
        extension (str) defaults to "all" : the extension of the files to search for
            can be any type of extension file (e.g. ".dat", or ".csv") or "all" meaning no filter is made on the files being returned
    Retuns:
        list[str] : the list of files paths ordered or not
    """
    files_ = [
        os.path.join(folder_to_search, f)
        for f in os.listdir(folder_to_search)
        if os.path.isfile(os.path.join(folder_to_search, f))
    ]
    files_filtered = filter_out_system_files(files_)

    if extension != "all":
        files_filtered = [file for file in files_filtered if file.endswith(extension)]

    if ordered_by_id:
        if files_filtered[0].endswith(".parquet") or files_filtered[0].endswith(".csv"):
            dict_files = {
                get_accelero_id_from_parquet_or_csv_files(file): file
                for file in files_filtered
            }
        files_filtered = [dict_files[key] for key in sorted(dict_files.keys())]

    return files_filtered


###############################################################################
#                        STANDARDIZE LOADING FILES                            #
###############################################################################


def load_parquet_or_csv(path_: str, verbose: bool = False) -> pd.DataFrame:
    """
    Load the parquet or csv file and return the dataframe.

    Args:
        path (str): the path to the parquet or csv file
    Returns:
        pd.DataFrame: the dataframe loaded
    """
    try:
        os.path.exists(path_)
    except FileNotFoundError:
        raise ValueError(f"The file {path_} doesn't exist")

    if path_.endswith(".parquet"):
        df = pd.read_parquet(path_, engine="pyarrow")
    elif path_.endswith(".csv"):
        df = pd.read_csv(path_)
    else:
        raise ValueError(f"The file {path_} is not a parquet or csv file")

    if verbose:
        print(f"    The file {path_.split(os.sep)[-1]} has been loaded successfully")
    return df


def load_all_sensor_files(
    path_to_dataset: str, subset: str, ordered: bool = False, verbose: bool = False
) -> dict:
    """
    Load all the files from sensors in the dataset given limits.
    ex : load_all_sensor_files("/Volumes/DATAJOJO/20240319 - Cordemais - Marc Doceul/DATA INGESTION","from 30 to 40")
    will load all the sensor files in the folder from the 30th to the 40th file.

    Args:
        path_to_dataset (str): the path to the dataset
        subset (str): the subset you want to load :
            - "all" : all the files
            - "from 30 to 40" : the files from 30 to 40
        ordered (bool, optional): returns a dict ordered from accelero id. Defaults to False.
        verbose (bool, optional): if True, the function will print infos. Defaults to False.
    Returns:
        dict: a dictionnary with the following structure :
            {"sensor_id1" : pd.DataFrame, "sensor_id2" : pd.DataFrame ...}
            with sensor id ranked alphabetically if ordered is selected
    """
    all_sensor_files = get_all_files_within_one_folder(
        path_to_dataset, True, extension=".parquet"
    )
    if verbose:
        print(len(all_sensor_files), all_sensor_files)

    if subset == "all":
        pass
    elif "from" in subset:
        start, stop = subset.split("from ")[1].split(" to ")
        print(start, stop)
        all_sensor_files = all_sensor_files[int(start) : int(stop)]

    all_accel = {}
    for accel_path in tqdm(all_sensor_files, desc="Loading all sensor data"):
        accelero_id = get_accelero_id_from_parquet_or_csv_files(accel_path)
        try:
            accelero_data = load_parquet_or_csv(accel_path)
            all_accel[accelero_id] = accelero_data
        except OSError:
            print("Error with the file : ", accel_path)
            continue
        gc.collect()

    return all_accel
