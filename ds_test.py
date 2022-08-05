"""
This file tests the data structure that reads and parses all files
"""
# %% 
import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
import scipy

import settings
from blinkit import io, viewers
from blinkit import data as data
from blinkit import blink_params
from blinkit import blink_fit, plot, sp_filters
import os
from blinkit import *
# %%
bf = BlinkFrame()
for i in range(10):
    current_filename = f"Device_{i}_Volts.xls"
    dir = ['Chandrika_Yadav', 'ST']
    file_path = f"{settings.PATH_TO_DATA}/21-06-22/{dir[1]}/{current_filename}"
    if os.path.exists(file_path):
        print(f"{current_filename} exists")
        bf.add_file(file_path, tag = f'{current_filename}')
        #df_eog, head = io.read_Voltsxls(file_path, include_header=True)
        #info_dict = interpret_header(head)
        #print(f"tag {info_dict['tag']}")
    else:
        print(f"{current_filename} does not exist")
# %%
eog, df, labels = bf.label_blinks()
# %%
import os
from blinkit import io
# %%
path_to_data = "../ADS1299+Video Blink data"
# %%
blink_files = []
joinpath = lambda x, y: f"{x}/{y}"
def give_files(path, extension = '.xls'):
    filter_files = []
    for dirname, folders, files in os.walk(path):
        for file in files: 
            if file.endswith(extension):
                filter_files.append(joinpath(dirname, file))

        for folder in folders:
            filter_files.extend(give_files(folder, extension))
    return filter_files
# %%
files = give_files(path_to_data)
# %%
file_names = []
for file in files:
    try:
        df_eog, head = io.read_Voltsxls(file, include_header=True)
        info_dict = viewers.interpret_header(head)
        file_names.append(info_dict['tag'])
    except ValueError as e:
        print("Error at", file)
        #print(e)
# %%
match = np.vectorize(re.match)
f = open(file, 'r')
file_aslist = [line.rstrip('\n') for line in f]
table_start = np.where(match('CH\d', file_aslist) != None)[0]
# %%
