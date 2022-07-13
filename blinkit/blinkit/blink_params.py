"""
This file holds the methods for aggregating the different base parameters of a blink
The following characteristics are found for each blinks in the following methods - 
    - start, end, width (duration)
    - duration at specific relative heights ([0.2, 0.5, 0.9])
    - location of maximum closing speed, maximum opening speed
    - blink interval to next/ previous
    - moment characteristics of blinks as distributions

The following characteristics are found per episode
    - Blink frequency per episode -> # blinks/ duration (in seconds)
"""

import pandas as pd
from scipy.signal import peak_widths, savgol_filter
from scipy.stats import moment
import numpy as np

class blink_stats:

    def perform(eog_data: list, blinks_df: pd.DataFrame):
        blink_stats.calc_widths(eog_data, blinks_df)
        blink_stats.calc_derivs(eog_data, blinks_df)
        blink_stats.blink_intervals(eog_data, blinks_df)
        blink_stats.get_moments(eog_data, blinks_df)
    def calc_widths(eog_data: list, blinks_df: pd.DataFrame, rel_values: list = [0, 0.2, 0.5, 0.95]):
        for val in rel_values:
            width, width_vals, b, c = peak_widths(eog_data, blinks_df["Peaks"], rel_height = 1 - val)
            blinks_df[f"{val}width"] = width
    
    def calc_derivs(eog_data: list, blinks_df: pd.DataFrame, window_length: int = 3, polyorder: int =2):
        """
        Perform the derivative calculation for blinks through `scipy.signal.savgol_filter` \n
        The respective columns added in this function are:
            - `spmax_close`: array index of max derivative while closing of eyelids
            - `spmin_open`: array index of min derivative while opening of eyelids
            - Something with second derivative?
        """

        first_deriv = savgol_filter(eog_data, window_length=window_length, polyorder=polyorder, deriv = 1)
        second_deriv = savgol_filter(eog_data, window_length=window_length, polyorder=polyorder, deriv = 2)

        spmax = []
        spmin = []
        for count, blink in blinks_df.iterrows():
            start = blink["Onsets"]
            end = blink["Offsets"]
            if np.isnan(start) or np.isnan(end):
                spmax.append(np.nan)
                spmin.append(np.nan)
                continue
            start = int(start)
            end = int(end)

            first_deriv_part = first_deriv[start: end]
            sec_deriv_part = second_deriv[start:end]
            spmax.append(np.where(first_deriv_part == np.max(first_deriv_part))[0][0])
            spmin.append(np.where(first_deriv_part == np.min(first_deriv_part))[0][0])

        blinks_df["spmax_close"] = pd.Series(spmax, dtype = "Int64")
        blinks_df['spmin_open'] = pd.Series(spmin, dtype = "Int64")
    
    def blink_intervals(eog_data: list, blinks_df: pd.DataFrame, int_type = 'forward'):
        """
        Track the array distance between blinks. \n
        Option `int_type` list the interval type -> `{'forward', 'backward'}`
        """
        assert int_type == 'forward' or int_type == 'backward', f"Incorrect interval type {int_type}"
        blink_intervals = []

        for count, blink in blinks_df.iterrows():
            if int_type == 'forward':
                if count + 1 == len(blinks_df):
                    blink_intervals.append(np.nan)
                    continue
                blink_intervals.append(blinks_df['Onsets'][count + 1] - blinks_df['Offsets'][count])


            elif int_type == 'backward':
                if count == 0:
                    blink_intervals.append(np.nan)
                    continue
                blink_intervals.append(blinks_df['Onsets'][count] - blinks_df['Offsets'][count - 1])
        blinks_df["blink_interval"] = pd.Series(blink_intervals, dtype = "Int64")
    
    def get_moments(eog_cleaned: list, blinks_df: pd.DataFrame, moments: list = [1, 2, 3, 4]):
        return_vals = []
        for count, blink in blinks_df.iterrows():
            start = blink["Onsets"]
            end = blink["Offsets"]
            if np.isnan(start) or np.isnan(end):
                return_vals.append([np.nan] * len(moments))
                continue
            start = int(start)
            end = int(end)
            blink_range = eog_cleaned[start: end]
            curr_blink_moments = []
            for val in moments:
                val_moment = moment(blink_range, moment = val, nan_policy = 'omit')
                curr_blink_moments.append(val_moment)
            return_vals.append(curr_blink_moments)
        new_df = pd.DataFrame(return_vals, columns = [f"moment{val}" for val in moments])
        for i in new_df.columns:
            blinks_df[i] = new_df[i]
