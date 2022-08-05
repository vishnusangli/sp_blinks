# %%
import pandas as pd
import numpy as np

import settings
import matplotlib.pyplot as plt
from blinkit import data, sp_filters, io
from blinkit import plot as myplot

from scipy.signal import detrend
from scipy.signal import savgol_filter
from scipy import signal

np.set_printoptions(precision=8)
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
cutoff = 6.3
fs = 256
filtered_data = sp_filters.butter_lowpass_filter(use_data, cutoff, fs, order = 2)
# %%
cutoff=0.05
b, a = signal.butter(2, cutoff, btype='lowpass') #low pass filter
filtered_data= signal.filtfilt(b, a, use_data)
#filtered_data = data.butter_lowpass_filter(use_data, 0.005, 100, 2)
#filtered_data = data.butter_highpass_filter(filtered_data, 0.05, 1000, 2)

fig, ax = plt.subplots(1,3, figsize = (10, 3))
ax[0].plot(filtered_data, label = 'raw')
ax[0].set_title("Raw")
ax[0].grid()

first_deriv = savgol_filter(filtered_data, window_length= 31, polyorder= 2, deriv = 1)
ax[1].plot(first_deriv)
ax[1].set_title("First Deriv")
ax[1].grid()

second_deriv = savgol_filter(filtered_data, 51, 2, 2)
ax[2].plot(second_deriv)
ax[2].set_title("Second Deriv")
ax[2].grid()
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
detrend_arr = data.padded_interp(filtered_data, minimas)
dtd_data = filtered_data - detrend_arr
plt.plot(detrend(filtered_data), alpha = 0.5)
plt.plot(dtd_data, alpha = 0.5)

# %%
#### NORMALIZATION ####
dtd_data = np.divide(dtd_data, np.max(dtd_data))
plt.plot(dtd_data)
# %%
#### SEGMENTATION ####
plt.figure(figsize=(10, 8))
plt.plot(dtd_data)
zero_vals = np.where(data.approx_equal(dtd_data, 0, e = 1e-2))[0]
elems_arr, pts_arr = data.exclude_points(dtd_data, zero_vals, window_threshold=1)
plt.scatter(zero_vals, data.point_locate(dtd_data, zero_vals), alpha = 0.1)
for i in range(len(elems_arr)):
    plt.plot(elems_arr[i], pts_arr[i])
plt.yscale('log')
plt.grid()
# %%
### BLINK SELECTION ###
lims = data.complement_ends(dtd_data, zero_vals, window_threshold=10)

new_lims = []
for i in lims:
    if i[1] - i[0] > 20:
        new_lims.append(i)


#lims = new_lims
lims = [data.lims_change(dtd_data, i, e = 1e-2) for i in lims]
#lims = data.group_blinks(dtd_data, lims)
filt = data.filter_blinks(dtd_data, lims, threshold=0.2, duration = 900)
plt.figure(figsize=(10, 8))
plt.plot(dtd_data)
#plt.scatter(zero_vals, data.point_locate(dtd_data, zero_vals), alpha = 0.1)
for i in range(len(lims)):
    elems = np.arange(*lims[i])
    plt.plot(elems, dtd_data[elems], alpha = 0.5)
    #box = myplot.create_rect(dtd_data, lims[i], alpha = 0.3, fill = filt[i])
    #plt.gca().add_patch(box)

plt.grid()
og = detrend(filtered_data)
og = np.divide(og, np.max(og))
plt.plot(og, color = 'gray')
# %%
#### OVERLAP AND DISPLAY BLINKS TOGETHER ####
good_blinks = [] #Operating the filter on list
for i in range(len(lims)):
    if filt[i]:
        good_blinks.append(lims[i])
        plt.plot(*data.center_blink(dtd_data[lims[i][0]: lims[i][1]]))
plt.grid()
plt.yscale('log')
# %%
#### CONVERT BLINKS TO IMAGE ####
xlims = [-100, 100, 2]
ylims = [0, 1, 0.01]
print(np.divide(xlims[1] - xlims[0], xlims[2]), np.divide(ylims[1] - ylims[0], ylims[2]))
imgs = []
show = False
for i in range(len(good_blinks)):
    elem_arr, blink_arr = data.center_blink(dtd_data[good_blinks[i][0]: good_blinks[i][1]])
    a = myplot.convert_img(elem_arr, blink_arr, ylims = ylims,
                    xlims = xlims, fill = True)
    imgs.append(a)
    if show:
        plt.figure()
        plt.imshow(a)
        plt.colorbar()
    #plt.xticks(np.arange(xlims[0], xlims[1]))
imgs = np.array(imgs)
print(imgs.shape)
# %%
myplot.plot_blinks(imgs, name = current_filename, numplots = (4, 7))

# %%

# %%



