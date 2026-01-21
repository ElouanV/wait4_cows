#!/usr/bin/python
# -*- coding: utf-8 -*-

# creator : Joseph Allyndree
# date : 2024-12-12

import os
from datetime import timedelta
from pprint import pprint as pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

###############################################################################
#                 ACCELEROMETERS PLOTTING FUNCTIONS                           #
###############################################################################


def cut_accelero_data(
    accelero_data_: pd.DataFrame,
    lower_date: pd.DatetimeIndex = 0,
    upper_date: pd.DatetimeIndex = 0,
) -> pd.DataFrame:
    """
    Cut the accelerometer data for a specific time range

    Args:
        accelero_data_ (pd.DataFrame): the accelerometer data
        lower_date (pd.DatetimeIndex): the lower limit date to cut the accelerometer to
        upper_date (pd.DatetimeIndex): the upper limit date to cut the accelerometer to

    Returns:
        pd.DataFrame: the cut accelerometer data
    """
    accelero_data = accelero_data_.copy()
    short_accelero_data = accelero_data[accelero_data["relative_DateTime"] < upper_date]
    if short_accelero_data.empty:
        short_accelero_data_2 = accelero_data[
            accelero_data["relative_DateTime"] > lower_date
        ]
        short_accelero_data_2 = short_accelero_data_2[
            short_accelero_data_2["relative_DateTime"] < upper_date
        ]
        return short_accelero_data_2

    else:
        short_accelero_data = short_accelero_data[
            short_accelero_data["relative_DateTime"] > lower_date
        ]
        return short_accelero_data


def plot_accelerometer(
    accelero_data: pd.DataFrame,
    column_x_axis: str,
    accelero_id: str,
    data_collection_date: str,
    lower_date: pd.DatetimeIndex = 0,
    upper_date: pd.DatetimeIndex = 0,
    save: bool = False,
    save_folder: str = "./",
):
    """
    Plot the accelerometer data for a specific time range

    Args:
        accelero_data (pd.DataFrame): the accelerometer data
        column_x_axis (str): the column to use for the x axis
        accelero_id (str): the accelerometer id
        lower_date (pd.DatetimeIndex, optional): the lower limit date to plot. Defaults to 0.
        upper_date (pd.DatetimeIndex, optional): the upper limit date to plot. Defaults to 0.
        save (bool, optional): whether to save the plot or not. Defaults to False.
            If True, the plot will be saved in the save_folder with the name
            "<data_collection_date>_<accelero_id>_accelero.png"
        save_folder (str, optional): the folder to save the plot. Defaults to "./"

    Returns:
        plot
    """
    plt.close("all")
    plt.figure(figsize=(31, 14))
    plt.rcParams.update({"font.size": 20})
    plt.tight_layout()

    if upper_date == 0 or lower_date == 0:
        short_accelero_data = accelero_data.copy()
    else:
        short_accelero_data = cut_accelero_data(accelero_data, lower_date, upper_date)

    magnitude = np.sqrt(
        short_accelero_data["acc_x"] ** 2
        + short_accelero_data["acc_y"] ** 2
        + short_accelero_data["acc_z"] ** 2
    )

    plt.plot(
        short_accelero_data[column_x_axis],
        magnitude,
        alpha=0.6,
        linewidth=0.5,
        label=accelero_id,
    )

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper right")
    plt.grid()

    plt.title(
        f"Acceleration data for {accelero_id} - {column_x_axis} - {lower_date} to {upper_date}",
        wrap=True,
    )

    plt.xlabel(column_x_axis)
    x_ticks = pd.date_range(
        start=lower_date, end=upper_date, freq="D"
    )  # Daily frequency
    plt.xticks(x_ticks, rotation=45, fontsize=17)
    plt.ylabel("Magnitude (g)")

    if save:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        name_fig = f"{data_collection_date}_{accelero_id}_accelero.png"
        plt.savefig(os.path.join(save_folder, name_fig))

    return plt.show()


def plot_2020_accelerometer(
    accelero_data_: pd.DataFrame,
    accelero_id: str,
    days_to_add: int,
    lower_date: pd.DatetimeIndex = 0,
    upper_date: pd.DatetimeIndex = 0,
):
    """
    Plot the accelerometer data for a specific time range
    Args:
        accelero_data_ (pd.DataFrame): the accelerometer data
        accelero_id (str): the accelerometer id
        days_to_add (int): the number of days to add to make it 2024 again
            Can do the calculation with this website : https://www.timeanddate.com/date/durationresult.html?d1=01&m1=01&y1=2020&d2=23&m2=07&y2=2024
        lower_date (pd.DatetimeIndex, optional): the lower limit date to plot. Defaults to 0.
        upper_date (pd.DatetimeIndex, optional): the upper limit date to plot. Defaults to 0.

    Returns:
        plot
    """
    plt.close("all")
    plt.figure(figsize=(14, 8))
    plt.margins(x=0)
    accelero_data = accelero_data_.copy()
    accelero_data["relative_DateTime"] = accelero_data["relative_DateTime"] + timedelta(
        days=days_to_add
    )

    if upper_date == 0 or lower_date == 0:
        short_accelero_data = accelero_data.copy()
        magnitude = np.sqrt(
            accelero_data["acc_x"] ** 2
            + accelero_data["acc_y"] ** 2
            + accelero_data["acc_z"] ** 2
        )

    else:
        short_accelero_data = cut_accelero_data(accelero_data, lower_date, upper_date)
        magnitude = np.sqrt(
            short_accelero_data["acc_x"] ** 2
            + short_accelero_data["acc_y"] ** 2
            + short_accelero_data["acc_z"] ** 2
        )

    plt.plot(
        short_accelero_data["relative_DateTime"],
        magnitude,
        alpha=0.6,
        label=accelero_id,
    )
    plt.legend(loc="upper right")
    plt.grid()
    plt.title(
        "Accelerometer data for the time range : "
        + str(lower_date)
        + " to "
        + str(lower_date)
        + " for the accelerometer "
        + accelero_id,
        wrap=True,
    )
    plt.xlabel("relative_DateTime")
    plt.ylabel("Magnitude")
    return plt.show()


def plot_multiple_accelerometers(
    all_accel_data: dict,
    accel_to_plot_ids: list[str],
    lower_date: pd.DatetimeIndex = 0,
    upper_date: pd.DatetimeIndex = 0,
    ref_sensor_id: str = "",
):
    """
    Plot multiple accelerometers data
    Args:
        all_accel_data (dict): the datastructure containing all the accelerometer data with the accelerometer id as key
        accel_to_plot_ids (list[str]): the list of the accelerometers ids to plot
        lower_date (pd.DatetimeIndex, optional): the lower limit date to plot. Defaults to 0.
        upper_date (pd.DatetimeIndex, optional): the upper limit date to plot. Defaults to 0.
        ref_sensor_id (str, optional): the reference accelerometer id. Defaults to "" (empty string).

    Returns:
        plot
    """
    plt.close("all")
    plt.figure(figsize=(14, 8))
    plt.margins(x=0)

    all_accel_data_to_plot = {
        sensor_id: all_accel_data[sensor_id]
        for sensor_id in accel_to_plot_ids
        if sensor_id in all_accel_data
    }
    # print("all_accel_data_to_plot", all_accel_data_to_plot)

    for sensor_id, accelero_data_ in tqdm(
        all_accel_data_to_plot.items(), desc="Plotting magnitude"
    ):
        if sensor_id not in accel_to_plot_ids or sensor_id == ref_sensor_id:
            continue
        accelero_data = accelero_data_.copy()
        if lower_date == 0 or upper_date == 0:
            short_accelero_data = accelero_data.copy()
            magnitude = np.sqrt(
                accelero_data["acc_x"] ** 2
                + accelero_data["acc_y"] ** 2
                + accelero_data["acc_z"] ** 2
            )
        else:
            short_accelero_data = cut_accelero_data(
                accelero_data, lower_date, upper_date
            )
            magnitude = np.sqrt(
                short_accelero_data["acc_x"] ** 2
                + short_accelero_data["acc_y"] ** 2
                + short_accelero_data["acc_z"] ** 2
            )
        plt.plot(
            short_accelero_data["relative_DateTime"],
            magnitude,
            alpha=0.4,
            label=sensor_id,
        )

    if ref_sensor_id != "":  # Adding the reference accelerometer
        if lower_date == 0 or upper_date == 0:
            short_accelero_data = all_accel_data[ref_sensor_id].copy()
            magnitude = np.sqrt(
                short_accelero_data["acc_x"] ** 2
                + short_accelero_data["acc_y"] ** 2
                + short_accelero_data["acc_z"] ** 2
            )
        else:
            short_accelero_data = cut_accelero_data(
                all_accel_data[ref_sensor_id], lower_date, upper_date
            )
            magnitude = np.sqrt(
                short_accelero_data["acc_x"] ** 2
                + short_accelero_data["acc_y"] ** 2
                + short_accelero_data["acc_z"] ** 2
            )
        plt.plot(
            short_accelero_data["relative_DateTime"],
            magnitude,
            alpha=0.4,
            label=ref_sensor_id,
        )

    plt.legend(loc="upper right")
    plt.grid()
    if lower_date == 0 or upper_date == 0:
        plt.title(
            "Accelerometer data for the whole time range : for the accelerometers "
            + str(accel_to_plot_ids),
            wrap=True,
        )
    else:
        plt.title(
            str(lower_date)
            + " to "
            + str(lower_date)
            + " for the accelerometers "
            + str(accel_to_plot_ids),
            wrap=True,
        )
    plt.xlabel("relative_DateTime")
    plt.ylabel("Magnitude")
    plt.plot()
    return plt.show()


def plot_multiple_2020_accelerometers(
    all_accel_data: dict,
    accel_to_plot_ids: list[str],
    days_to_add: int,
    seconds_to_add: int,
    lower_date: pd.DatetimeIndex = 0,
    upper_date: pd.DatetimeIndex = 0,
    ref_sensor_id: str = "",
):
    """
    Plot multiple accelerometers data
    Args:
        all_accel_data (dict): the datastructure containing all the accelerometer data with the accelerometer id as key
        accel_to_plot_ids (list[str]): the list of the accelerometers ids to plot
        days_to_add (int): the number of days to add to make it 2024 again
            Can do the calculation with this website : https://www.timeanddate.com/date/durationresult.html?d1=01&m1=01&y1=2020&d2=23&m2=07&y2=2024
        seconds_to_add (int): the number of seconds to add

        lower_date (pd.DatetimeIndex, optional): the lower limit date to plot. Defaults to 0.
        upper_date (pd.DatetimeIndex, optional): the upper limit date to plot. Defaults to 0.
        ref_sensor_id (str, optional): the reference accelerometer id. Defaults to "" (empty string).

    Returns:
        plot
    """
    plt.close("all")
    plt.figure(figsize=(14, 8))
    plt.margins(x=0)

    all_accel_data_to_plot = {
        sensor_id: all_accel_data[sensor_id]
        for sensor_id in accel_to_plot_ids
        if sensor_id in all_accel_data
    }

    for sensor_id, accelero_data_ in tqdm(
        all_accel_data_to_plot.items(), desc="Plotting magnitude"
    ):
        if sensor_id not in accel_to_plot_ids or sensor_id == ref_sensor_id:
            continue
        accelero_data = accelero_data_.copy()

        # check for the ["relative_DateTime_aligned"] ==> 2020 accelerometers
        if "relative_DateTime_aligned" in accelero_data.columns:
            accelero_data["relative_DateTime"] = accelero_data[
                "relative_DateTime_aligned"
            ].copy()

        else:  # 2024 accelerometers
            accelero_data["relative_DateTime"] = accelero_data[
                "relative_DateTime"
            ] + timedelta(days=days_to_add)
            accelero_data["relative_DateTime"] = accelero_data[
                "relative_DateTime"
            ] + timedelta(seconds=float(seconds_to_add))

        if lower_date == 0 or upper_date == 0:
            short_accelero_data = accelero_data.copy()
            magnitude = np.sqrt(
                accelero_data["acc_x"] ** 2
                + accelero_data["acc_y"] ** 2
                + accelero_data["acc_z"] ** 2
            )
        else:
            short_accelero_data = cut_accelero_data(
                accelero_data, lower_date, upper_date
            )
            magnitude = np.sqrt(
                short_accelero_data["acc_x"] ** 2
                + short_accelero_data["acc_y"] ** 2
                + short_accelero_data["acc_z"] ** 2
            )
        plt.plot(
            short_accelero_data["relative_DateTime"],
            magnitude,
            alpha=0.4,
            label=sensor_id,
        )

    if ref_sensor_id != "":  # Adding the reference accelerometer
        if lower_date == 0 or upper_date == 0:
            short_accelero_data = all_accel_data[ref_sensor_id].copy()
            magnitude = np.sqrt(
                short_accelero_data["acc_x"] ** 2
                + short_accelero_data["acc_y"] ** 2
                + short_accelero_data["acc_z"] ** 2
            )
        else:
            short_accelero_data = cut_accelero_data(
                all_accel_data[ref_sensor_id], lower_date, upper_date
            )
            magnitude = np.sqrt(
                short_accelero_data["acc_x"] ** 2
                + short_accelero_data["acc_y"] ** 2
                + short_accelero_data["acc_z"] ** 2
            )
        plt.plot(
            short_accelero_data["relative_DateTime"],
            magnitude,
            alpha=0.4,
            label=ref_sensor_id,
        )

    print(accelero_data.head())
    plt.legend(loc="upper right")
    plt.grid()
    plt.title(
        str(lower_date)
        + " to "
        + str(lower_date)
        + " for the accelerometers "
        + str(accel_to_plot_ids),
        wrap=True,
    )
    plt.xlabel("relative_DateTime")
    plt.ylabel("Magnitude")
    plt.plot()
    return plt.show()


###############################################################################
#                       RSSI PLOTTING FUNCTIONS                               #
###############################################################################


def plot_multiple_rssi(
    all_rssi_data: dict,
    accel_to_plot_ids: list[str],
    lower_date: pd.DatetimeIndex = 0,
    upper_date: pd.DatetimeIndex = 0,
):
    """
    Plot multiple accelerometers data
    Args:
        all_rssi_data (dict): the datastructure containing all the accelerometer data with the accelerometer id as key
        accel_to_plot_ids (list[str]): the list of the accelerometers ids to plot
        lower_date (pd.DatetimeIndex, optional): the lower limit date to plot. Defaults to 0.
        upper_date (pd.DatetimeIndex, optional): the upper limit date to plot. Defaults to 0.

    Returns:
        plot
    """
    plt.close("all")
    plt.figure(figsize=(14, 8))
    plt.margins(x=0)

    all_rssi_data_to_plot = {
        sensor_id: all_rssi_data[sensor_id]
        for sensor_id in accel_to_plot_ids
        if sensor_id in all_rssi_data
    }

    for sensor_id, rssi_data_ in tqdm(
        all_rssi_data_to_plot.items(), desc="Plotting RSSI"
    ):
        if sensor_id not in accel_to_plot_ids:
            continue
        rssi_data_1 = rssi_data_.copy()
        if lower_date == 0 or upper_date == 0:
            short_rssi_data = rssi_data_1.copy()
        else:
            short_rssi_data = cut_accelero_data(rssi_data_1, lower_date, upper_date)

        plt.scatter(
            short_rssi_data["relative_DateTime"],
            short_rssi_data["RSSI"],
            alpha=0.4,
            label=sensor_id,
        )

    plt.legend(loc="upper right")
    plt.grid()
    if lower_date == 0 or upper_date == 0:
        plt.title(
            "RSSI data for the whole time range for the accelerometers "
            + str(accel_to_plot_ids),
            wrap=True,
        )
    else:
        plt.title(
            "RSSI data from "
            + str(lower_date)
            + " to "
            + str(lower_date)
            + " for the accelerometers "
            + str(accel_to_plot_ids),
            wrap=True,
        )
    plt.xlabel("relative_DateTime")
    plt.ylabel("RSSI (dbm)")
    plt.plot()
    return plt.show()


def plot_single_sensor_separate_rssi(
    single_rssi_data: pd.DataFrame,
    number_of_plots: int,
    accelero_id: str,
    data_collection_date: str,
    accel_to_plot_ids: list[str] | str,
    lower_date: pd.DatetimeIndex = 0,
    upper_date: pd.DatetimeIndex = 0,
    column_x_axis: str = "glob_sensor_DateTime",
    save: bool = False,
    save_folder: str = "./",
):
    """
    Plot the RSSI data of a sensor with the other detected data
    Args:
        single_rssi_data (pd.DataFrame): the RSSI data for a single sensor
        number_of_plots (int): the number of plots to draw (the more the number of plots, the
            less number of accelerometers per plot)
        accelero_id (str): the accelerometer id
        data_collection_date (str): the date of the data collection (from the .env file)
        accel_to_plot_ids (list[str]): the list of the accelerometers ids to plot
        lower_date (pd.DatetimeIndex, optional): the lower limit date to plot. Defaults to 0.
        upper_date (pd.DatetimeIndex, optional): the upper limit date to plot. Defaults to 0.
        column_x_axis (str, optional): the column to use for the x axis. Defaults to "glob_sensor_DateTime".
        save (bool, optional): whether to save the plot or not. Defaults to False.
            If True, the plot will be saved in the save_folder with the name
            "<data_collection_date>_<accelero_id>_RSSI.png"
        save_folder (str, optional): the folder to save the plot. Defaults to "./"
    Returns:
        plot
    """
    plt.close("all")
    plt.figure(figsize=(14, 8))
    plt.tight_layout()

    # a 1 x number_of_plots grid of plots
    fig, axs = plt.subplots(number_of_plots, 1, figsize=(30, 12 * number_of_plots))
    plt.rcParams.update({"font.size": 20})

    rssi_data_1 = single_rssi_data.copy()
    if lower_date == 0 or upper_date == 0:
        short_rssi_data = rssi_data_1.copy()
    else:
        short_rssi_data = cut_accelero_data(rssi_data_1, lower_date, upper_date)

    unique_accelero = list(short_rssi_data["accelero_id"].unique())
    if accel_to_plot_ids == "all":
        accelero_to_plot = unique_accelero
    else:
        accelero_to_plot = [
            accelero_id
            for accelero_id in accel_to_plot_ids
            if accelero_id in unique_accelero
        ]
    splitted_id_accelero = np.array_split(accelero_to_plot, number_of_plots)

    for i, ax in enumerate(axs.flat):
        for ble_id in splitted_id_accelero[i]:
            rssi_tab_ble = short_rssi_data[short_rssi_data["accelero_id"] == ble_id]
            ax.plot(
                rssi_tab_ble[column_x_axis],
                rssi_tab_ble["RSSI"],
                label=ble_id,
                linewidth=0.5,
            )
            ax.set_xlabel(column_x_axis)
            ax.set_ylabel("RSSI (dBm)")
            ax.set_title(
                f"{data_collection_date} - RSSI data for accelerometer {accelero_id} - Column {column_x_axis}",
                wrap=True,
            )
            ax.legend(loc="upper right")

    plt.xlabel("relative_DateTime")
    plt.ylabel("RSSI (dbm)")
    plt.plot()

    if save:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        name_fig = f"{data_collection_date}_{accelero_id}_RSSI.png"
        plt.savefig(os.path.join(save_folder, name_fig))

    return plt.show()
