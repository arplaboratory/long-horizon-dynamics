import numpy as np
import pandas as pd
from utils.ulog_tools import pandas_from_topic
from utils.quat_utils import slerp
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt


def compute_flight_time(data_df):
    """
    The flight time will be determined based on the 'landed' topic of the ulog to only consider actual flight during identification.

    Depending on the number of sections where the landed topic is 1, the following cases are distinguished:
    - No groups with landed = 1: No flight detected. Start and end time of the flight will be returned as zero
    - One group with landed = 1: The start and end time of the entire flight log will be returned
    - Even number of groups with landed = 1: An array with start and end time of each detected flight segment will be returned. If multiple flight segments were detected, a warning is issued.
    - Odd number of groups with landed = 1: An array with start and end time of each detected flight segment will be returned. If multiple flight segments were detected, a warning is issued.
    """
    print("\nComputing flight time...")
    landed_groups = data_df.groupby(
        (data_df["landed"].shift() != data_df["landed"]).cumsum()
    )
    num_of_groups = landed_groups.ngroups

    # single group detected - flight either started and ended outside of log of aircraft did not fly at all
    if num_of_groups == 1:
        if landed_groups.get_group(1)["landed"].iloc[0] == 1:
            print("No flight detected. Please check the landed state.")
            return [{"t_start": 0, "t_end": 0}]
        else:
            pass

    # multiple groups detected - flight started and/or ended within the flight log duration
    else:
        if num_of_groups % 2 == 1 and num_of_groups > 3:
            print(
                "WARNING: More than one flight detected. The start and end times of the individual segments will be returned."
            )
        if num_of_groups % 2 == 0 and num_of_groups > 2:
            print(
                "WARNING: More than one flight detected. The start and end times of the individual segments will be returned."
            )

        # find flight segments with landed topic = 0 and add the corresponding start and end times to the flight_times array
        flight_times = []
        start_index = -1
        if landed_groups.get_group(1)["landed"].iloc[0] == 1:
            start_index = 2
        else:
            start_index = 1

        for i in range(start_index, num_of_groups + 1, 2):
            flight_segment = landed_groups.get_group(i)[["timestamp", "landed"]]
            flight_times.append(
                {
                    "t_start": flight_segment.iloc[0, 0],
                    "t_end": flight_segment.iloc[-1, 0],
                }
            )
        return flight_times

    print("Flight time computation completed successfully.")
    return [{"t_start": act_df.iloc[0, 0], "t_end": act_df.iloc[-1, 0]}]



def butter_lowpass(cutoff, fs, order=5):
    """
    Design a Butterworth low-pass filter.
    
    Inputs:
    - cutoff: The cutoff frequency of the filter (Hz)
    - fs: The sampling frequency of the data (Hz)
    - order: The order of the filter (higher values provide a steeper roll-off)
    
    Returns:
    - b, a: Numerator and denominator polynomials of the IIR filter.
    """
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalized cutoff frequency
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def applyButterLowpassFilter(data, cutoff, fs, order=5):
    """
    Apply a Butterworth low-pass filter to the data.
    
    Inputs:
    - data: The input signal (1D array)
    - cutoff: The cutoff frequency (Hz)
    - fs: The sampling frequency (Hz)
    - order: The order of the filter (default is 5)
    
    Returns:
    - The filtered signal (1D array)
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)  # Apply filter using filtfilt for zero-phase distortion
    return y


def moving_average(x, w=7):
    return np.convolve(x, np.ones(w), "valid") / w


# def filter_df(data_df, w=11):
#     data_np = data_df.to_numpy()
#     column_list = data_df.columns
#     new_df = pd.DataFrame()
#     for i in range(data_np.shape[1]):
#         new_df[column_list[i]] = moving_average(data_np[:, i])
#     return new_df

def filter_df(data_df, cutoff=5, fs=100, order=4):
    """
    Apply a Butterworth low-pass filter to each column of a DataFrame.
    
    Inputs:
    - data_df: Input pandas DataFrame with columns to be filtered.
    - cutoff: The cutoff frequency (Hz)
    - fs: The sampling frequency (Hz)
    - order: The order of the filter (default is 5)
    
    Returns:
    - A new DataFrame with filtered data.
    """
    data_np = data_df.to_numpy()
    column_list = data_df.columns
    new_df = pd.DataFrame()

    for i in range(data_np.shape[1]):
        # Apply the Butterworth low-pass filter to each column of the DataFrame
        filtered_column = applyButterLowpassFilter(data_np[:, i], cutoff, fs, order)
        new_df[column_list[i]] = filtered_column

    return new_df


def resample_dataframe_list(
    df_list, time_window=None, f_des=100.0, slerp_enabled=False, filter=True
):
    """create a single dataframe by resampling all dataframes to f_des [Hz]

    Inputs:     df_list : List of ulog topic dataframes to resample
                t_start : Start time in us
                t_end   : End time in us
                f_des   : Desired frequency of resampled data
    """
    if time_window is None:
        # select full ulog time range
        df = df_list[0]
        timestamp_list = df["timestamp"].to_numpy()
        t_start = timestamp_list[0]
        t_end = timestamp_list[-1]

    else:
        t_start = time_window["t_start"]
        t_end = time_window["t_end"]

    # compute desired Period in us to be persistent with ulog timestamps
    assert f_des > 0, "Desired frequency must be greater than 0"
    T_des = 1000000.0 / f_des

    n_samples = int((t_end - t_start) / T_des)
    res_df = pd.DataFrame()
    new_t_list = np.arange(t_start, t_end, T_des)
    for df in df_list:
        df = filter_df(df)
        df_end = df["timestamp"].iloc[[-1]].to_numpy()
        if df_end < t_end:
            t_end = int(df_end)

    for df in df_list:
        # use slerp interpolation for quaternions
        # add a better criteria than the exact naming at a later point.
        if "q0" in df and slerp_enabled:
            q_mat = slerp_interpolate_from_df(df, new_t_list[0])

            for i in range(1, len(new_t_list)):
                q_new = slerp_interpolate_from_df(df, new_t_list[i])
                q_mat = np.vstack((q_mat, q_new))
            attitude_col_names = list(df.columns)
            attitude_col_names.remove("timestamp")
            new_df = pd.DataFrame(q_mat, columns=attitude_col_names)

        else:
            new_df = pd.DataFrame()
            for col in df:
                # new_df[col] = np.interp(new_t_list, df.timestamp, df[col])

                # use cubic spline interpolation
                cs = CubicSpline(df["timestamp"], df[col])
                new_df[col] = cs(new_t_list)

        res_df = pd.concat([res_df, new_df], axis=1)
        res_df = res_df.loc[:, ~res_df.columns.duplicated()]

    return res_df


def slerp_interpolate_from_df(df, new_t):
    df_sort = df.iloc[(df["timestamp"] - new_t).abs().argsort()[:2]]
    df_timestamps = df_sort["timestamp"].values.tolist()
    t_ratio = (new_t - df_timestamps[0]) / (df_timestamps[1] - df_timestamps[0])
    df_sort = df_sort.drop(columns=["timestamp"])

    q_new = slerp(
        df_sort.iloc[0, :].to_numpy(),
        df_sort.iloc[1, :].to_numpy(),
        np.array([t_ratio]),
    )
    return q_new


def crop_df(df, t_start, t_end):
    """crop df to contain 1 elemnt before t_start and one after t_end.
    This way it is easy to interpolate the data between start and end time."""
    df_start = df[df.timestamp <= t_start].iloc[[-1]]
    df_end = df[df.timestamp >= t_end].iloc[[0]]

    df = df[df.timestamp >= int(df_start.timestamp.to_numpy())]
    df = df[df.timestamp <= int(df_end.timestamp.to_numpy())]
    return df