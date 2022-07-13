# %% 
from msilib.schema import Class
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
curr_blink = df_blinks.iloc[0]
blink_range = eog_cleaned[int(curr_blink["Onsets"]): int(curr_blink["Offsets"])]
norm_data = data.normalize(blink_range)
plt.plot(norm_data)
x = range(len(norm_data))
# %%
p0 = [0.0005, 0.003, -0.002, 0.0005, 0]
temp_func = blink_fit.fitfunc_wrapper(blink_fit.paper_func.func, p0)
plt.plot(norm_data)
plt.plot(temp_func(x))
# %%
import scipy
popt, pcov = scipy.optimize.curve_fit(blink_fit.paper_func.func, x, norm_data, p0 = p0)
temp_func = blink_fit.fitfunc_wrapper(blink_fit.paper_func.func, popt)
# %%
plt.plot(norm_data)
plt.plot(temp_func(x))
# %%
# %
# %%
popt, pcov, x, y = blink_fit.lsqr_fit(norm_data, blink_fit.paper_func, p0, sampling_rate = 1, centering = False)
# %%
plt.plot(x, y)
# %%
