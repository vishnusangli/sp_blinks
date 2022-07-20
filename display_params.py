"""
This file contains code that plots the infographic that labels the
different parameters calculated for each blink

"""
# %% 
import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt

import settings
from blinkit import io
from blinkit import data as data
from blinkit import blink_params

import scipy.signal as signal
# %%
current_filename = "Device_1_Volts.xls"
dir = ['Chandrika_Yadav', 'ST']
file_path = f"{settings.PATH_TO_DATA}/21-06-22/{dir[1]}/{current_filename}"
# %%
df_eog = io.read_Voltsxls(file_path)
# %%
use_data = df_eog["CH1"]
use_data = list(use_data[6000:30000])
# %%
cutoff=0.05
b, a = signal.butter(2, cutoff, btype='lowpass') #low pass filter
filtered_data= signal.filtfilt(b, a, use_data)
# %%
eog_cleaned = nk.eog_clean(filtered_data, sampling_rate=100, method='neurokit')
plt.plot(data.detrend_standardize( eog_cleaned), alpha =0.7, label = "Cleaned")
plt.plot(data.detrend_standardize( filtered_data), alpha =0.7, label = "Raw")
plt.legend()
plt.grid()
# %%
blinks = nk.signal_findpeaks(eog_cleaned, relative_height_min=0)
print(f"{len(blinks['Peaks'])} Blinks found")
df_blinks = pd.DataFrame(blinks)

# %%
blink_params.blink_stats.perform(eog_cleaned, df_blinks)
# %%
from blinkit import blink_fit
# %%
curr_blink = df_blinks.iloc[3]
blink_range = eog_cleaned[int(curr_blink["Onsets"]): int(curr_blink["Offsets"])]
norm_data = data.normalize(blink_range)
plt.plot(norm_data)
x = range(len(norm_data))
# %%
# %%


#### PLOTTING INFORMATION ###
plt.plot(blink_range)
import scipy
first_deriv = scipy.signal.savgol_filter(blink_range, window_length = 3, polyorder = 2, deriv = 2)
# %%
fig, ax = plt.subplots(2, 1, sharex = True, figsize = (5, 8), facecolor = 'gray')
plt.suptitle("Characteristics of a blink")

ax[0].plot(blink_range)
ax[0].set_title("Blink")
ax[0].set_xticks([])
ax[0].set_yticks([])

## Peak
peak_val = int(curr_blink["Peaks"])
start = int(curr_blink["Onsets"])
end = int(curr_blink["Offsets"])

ax[0].scatter(peak_val - start, eog_cleaned[peak_val])

## start, end, width
ends = [0, end - start -1]
ax[0].scatter(ends, blink_range[ends])


ax[0].hlines(0, *ends)

ax[1].plot(first_deriv, color = 'darkorange')
ax[1].set_title("First Derivative")
ax[1].set_xticks([])
ax[1].set_yticks([])

## Deriv


# %%
