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
from blinkit import blink_fit, plot

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
temp = ["Onsets", "Offsets"]
lims = np.array(df_blinks[temp].values, dtype = int)
plt.figure(figsize=(10, 8))
plt.plot(eog_cleaned)
#plt.scatter(zero_vals, data.point_locate(dtd_data, zero_vals), alpha = 0.1)
for i in range(len(lims)):
    if lims[i][0] < 0 or lims[i][1] > len(eog_cleaned):  continue
    elems = np.arange(*lims[i], dtype = int)
    plt.plot(elems, eog_cleaned[elems], alpha = 0.5, color =  'crimson')
    box = plot.create_rect(eog_cleaned, lims[i], alpha = 0.3, fill = False)
    plt.gca().add_patch(box)
plt.grid()
plt.xlim([0000, 5000])
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
blink_labels = blink_params.label_doubleblinks(df_blinks)
# %%
x = fits['param_0']
y = fits["param_1"]
z = fits['param_2']
blink_color = []
cols = ['red', 'blue', 'green']
for i in blink_labels:
    blink_color.append(cols[i])
plt.scatter(x, y, c = blink_color, alpha = 0.5)
# %%

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x,y,z, c=blink_color, marker="o" )
ax.set_xlim([0, 0.005])
ax.set_ylim3d([0.0005, -0.0005])
ax.set_zlim3d([-1, 1])
ax.set_ylabel("test")
# %%
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
# %%
Elbow_M = KElbowVisualizer(KMeans(), k = 10)
Elbow_M.fit(a)
Elbow_M.show()
# %%
AC = AgglomerativeClustering(n_clusters=4)
yhat_AC = AC.fit_predict(a)

# %%
from matplotlib import colors
cmap = colors.ListedColormap(["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"])
# %%
fig = plt.figure(figsize=(10,8))
ax = plt.subplot(111, projection='3d', label="bla")
ax.scatter(x, y, z, s=40, c=yhat_AC, marker='o', cmap = cmap)
ax.set_title("The Plot Of The Clusters")

ax.set_xlim([0, 0.005])
ax.set_ylim3d([0.0005, -0.0005])
ax.set_zlim3d([-1, 1])

plt.show()

# %%
blink_lims = []
for count, blink in df_blinks.iterrows():
    start = blink["Onsets"]
    end = blink["Offsets"]

    if pd.isna(start) or pd.isna(end):
        continue
    blink_lims.append([int(start), int(end)])

dtd_data = np.divide(eog_cleaned, np.max(eog_cleaned))
from scipy.signal import detrend
dtd_data = detrend(dtd_data)
# %%
filt = data.filter_blinks(dtd_data, blink_lims, threshold=0.2, duration = 900)
good_blinks = [] #Operating the filter on list
for i in range(len(blink_lims)):
    if filt[i]:
        good_blinks.append(blink_lims[i])
        plt.plot(*data.center_blink(dtd_data[blink_lims[i][0]: blink_lims[i][1]]))
plt.grid()
#plt.yscale('log')
# %%
new_blinks = [[1829, 1984],
 [2613, 2697],
 [4863, 5011],
 [5340, 5436],
 [5859, 5974],
 [6390, 6514],
 [6862, 6970],
 [8151, 8233],
 [9458, 9554],
 [10565, 10653],
 [11386, 11483],
 [12386, 12475],
 [13177, 13282],
 [14802, 14872],
 [16019, 16152],
 [18003, 18095],
 [19416, 19503],
 [20908, 20997],
 [22202, 22346]]
# %%
i = 6
j = 5
eog_list = filtered_data

plt.figure(figsize = (10, 8))
ax = plt.axes()
ax.set_facecolor("black")
plt.plot(*data.center_blink(eog_list[good_blinks[i][0]: good_blinks[i][1]]), alpha = 0.7, label = "neuro", color = "aquamarine")
plt.plot(*data.center_blink(eog_list[new_blinks[j][0]: new_blinks[j][1]]), alpha = 0.7, label = "mine", color = "yellow")
plt.legend()


plt.grid()
print(good_blinks[i])
print(new_blinks[j])
# %%
list(zip(good_blinks, new_blinks + []))
# %%
blinks = nk.signal_findpeaks(data.detrend_standardize(filtered_data), relative_height_min=0)
print(f"{len(blinks['Peaks'])} Blinks found")
df_blinks = pd.DataFrame(blinks)
# %%
"""
Use personal smoothening/data cleaning method
Apply neurokit's find_peaks
Filter blinks based on height, width, etc using filter_blinks
"""