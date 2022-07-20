# %% 
import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt

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
def label_doubleblinks(blinks_df: pd.DataFrame):
    """
    Label blinks based on whether they're double blinks

    Labeling schema:
        - `0`: single blink
        - `1`: first blink in double blink pair
        - `2`: second blink in double blink pair
    
    Arguments: 
        - `blinks_df`: `neurokit2`-generated blinks from eog data as a `pd.DataFrame`
    
    Returns:
        - `labels`: list of corresponding labels
    """
    labels = []
    for count, blink in blinks_df.iterrows():
        if not pd.isna(blink["post_interval"] ) and blink["post_interval"] == 0:
            labels.append(1)
        elif not pd.isna(blink["pre_interval"]) and blink["pre_interval"] == 0:
            labels.append(2)
        else:
            labels.append(0)
    return np.array(labels)

# %%
blink_labels = label_doubleblinks(df_blinks)
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
