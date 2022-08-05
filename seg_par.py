"""
This file holds the pipeline method for segmenting and characterizing blinks

"""
# %% 
import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
import scipy

import settings
from blinkit import io
from blinkit import data as data
from blinkit import blink_params
from blinkit import blink_fit, plot, sp_filters

from scipy.signal import detrend
import scipy.signal as signal
# %%
current_filename = "Device_9_Volts.xls"
dir = ['Chandrika_Yadav', 'ST']
file_path = f"{settings.PATH_TO_DATA}/21-06-22/{dir[1]}/{current_filename}"
# %%
df_eog, head = io.read_Voltsxls(file_path, include_header=True)
# %%
use_data = df_eog["CH1"]
use_data = list(use_data[100:-100])
# %%
filtered_data = sp_filters.butter_lowpass_filter(use_data, 25, 1000, order = 2)
# %%
#### SAVGOL METHOD FOR CHANGEPOINTS ####
minimas, maximas = data.ChangePointDetect.savgol_method(filtered_data, e = 1e-6)
plt.plot(filtered_data, label = 'data')
plt.scatter(minimas, data.point_locate(filtered_data, minimas), 
            label = 'minimas', alpha = 0.1, color = 'green')
plt.scatter(maximas, data.point_locate(filtered_data, maximas),
            label = 'maximas', alpha = 0.1, color ='orange')
plt.legend()
plt.grid()

 # %%
def moving_avg_minimas(eog_data, minimas, window = 5, diff = 1e-2):
    standardize_vals = []
    mv_avg = eog_data[0]

    for i in minimas:
        if abs(np.divide((eog_data[i] - mv_avg), mv_avg)) > diff:
            print("Here")
            continue
        else:
            print(abs(np.divide((eog_data[i] - mv_avg), mv_avg)))
            standardize_vals.append(i)
            window_start = max(0, i - window)
            mv_avg = np.average(eog_data[window_start:i + 1])
    return standardize_vals
a = moving_avg_minimas(filtered_data, minimas)
excluded = list(set(minimas) - set(a))
plt.plot(filtered_data, label = 'data')
plt.scatter(excluded, data.point_locate(filtered_data, excluded), 
            label = 'minimas', alpha = 0.1, color = 'green')
plt.legend()
plt.grid()
# %%
eog_cleaned = data.detrend_standardize(filtered_data)
plt.plot(eog_cleaned, alpha =0.7, label = "Cleaned")
plt.plot(data.normalize(detrend(filtered_data)), alpha =0.7, label = "Raw")
plt.legend()
plt.grid()
# %%
blinks = nk.signal_findpeaks(eog_cleaned, relative_height_min=0)
print(f"{len(blinks['Peaks'])} Blinks found")
df_blinks = pd.DataFrame(blinks)

# %%
blink_lims = []
for count, blink in df_blinks.iterrows():
    start = blink["Onsets"]
    end = blink["Offsets"]

    if pd.isna(start) or pd.isna(end):
        continue
    blink_lims.append([int(start), int(end)])

dtd_data = np.divide(eog_cleaned, np.max(eog_cleaned))
# %%
filt = data.filter_blinks(dtd_data, blink_lims, threshold=0.2, duration = 900)
good_blinks = [] #Operating the filter on list
for i in range(len(blink_lims)):
    if filt[i]:
        good_blinks.append(blink_lims[i])
        plt.plot(*data.center_blink(dtd_data[blink_lims[i][0]: blink_lims[i][1]]))
plt.grid()
#plt.yscale('log')
print(f"{len(good_blinks)} valid blinks")

# %%
def gen_blinktable(blink_eog: list, blink_lims: list[list[int]]) -> pd.DataFrame:
    """
    Given a list of blink slices, this method generates the starting 
    `neurokit`-like blink `pd.DataFrame`. It provides the following columns:
        - `"Onsets"`: Start of blink (eog array index)
        - `"Offsets"`: End of blink (eog array index)
        - `"Peak"`: blink peak (eog array index)
        - `"rise_height"`: Height of blink from start position. 
        - `"fall_height"`: Height of blink to end position.\n
    Height is calculated with the provided units in `blink_eog`. \n
    Arguments:
        - `blink_eog`: List of eog readings for the event
        - `blink_lims`: list of integer tuples containing `[start, end]` slices
        for blinks \n

    Returns: 
        - `blinks_df`: `pandas.DataFrame` table of blinks with information
    """
    blinks_arr = []
    for blink in blink_lims:
        start = int(blink[0])
        end = int(blink[1])
        blink_range = blink_eog[start:end]

        peak = np.where(blink_range == np.max(blink_range))[0][0]
        r_height = blink_range[peak] - blink_eog[start]
        f_height = blink_range[peak] - blink_eog[end]
        blinks_arr.append([start, end, peak + start, r_height, f_height])
    blinks_df = pd.DataFrame(blinks_arr, columns = ['Onsets', "Offsets", "Peak", "rise_height", "fall_height"])
    return blinks_df

# %%
df_blinks = gen_blinktable(eog_cleaned, good_blinks)
# %%
temp = ["Onsets", "Offsets"]
lims = np.array(df_blinks[temp].values, dtype = int)
plt.figure(figsize=(10, 8))
plt.plot(eog_cleaned)
#plt.scatter(zero_vals, data.point_locate(dtd_data, zero_vals), alpha = 0.1)
for i in range(len(lims)):
    if lims[i][0] < 0 or lims[i][1] > len(eog_cleaned):  continue
    elems = np.arange(*lims[i], dtype = int)
    plt.plot(elems, eog_cleaned[elems], alpha = 0.5, color =  'crimson')
    box = plot.create_rect(eog_cleaned, lims[i], alpha = 0.3, fill = True)
    plt.gca().add_patch(box)
plt.grid()
plt.xlim([5000, 15000])
#og = scipy.signal.detrend(filtered_data)
#og = np.divide(og, np.max(og))
#plt.plot(og, color = 'gray')
# %%
blink_params.blink_stats.perform(eog_cleaned, df_blinks)
# %%
fits = blink_fit.gen_fits(eog_cleaned, df_blinks, blink_fit.paper_func, p0 = None)

# %%
val = np.where(fits['sqr_dist'] == np.min(fits['sqr_dist']))[0][0]
fits.iloc[val, :]

plt.figure(figsize = (8, 5.5))
plot.plot_compare(eog_cleaned, fits, df_blinks, 31, blink_fit.paper_func)

plt.ylabel("EOG normalized")
plt.xlabel("Time (ms)")
plt.grid()
plt.title(f"Comparison of blink and fit for blink {val}")
# %%
"""
Use personal smoothening/data cleaning method
Apply neurokit's find_peaks
Filter blinks based on height, width, etc using filter_blinks
"""
filtered_data
eog_cleaned = data.detrend_standardize(filtered_data)


# %%
from blinkit import *

current_filename = "Device_4_Volts.xls"
dir = ['Chandrika_Yadav', 'ST']
file_path = f"{settings.PATH_TO_DATA}/21-06-22/{dir[1]}/{current_filename}"
# %%
bf = BlinkFrame()
# %%
bf.add_file(file_path)
# %%
eog, df, labels = bf.label_blinks()
# %%
df
# %%
current_filename = "Device_3_Volts.xlsx"
dir = ['Chandrika_Yadav', 'ST']
file_path = f"{settings.PATH_TO_DATA}/21-06-22/{dir[1]}/{current_filename}"

# %%
df_eog, head = io.read_Voltsxls(file_path, include_header=True)
df_eog.head()

# %%
