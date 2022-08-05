"""
This file provides an appropriate data structure and API for
accessing and viewing the blink data
"""
# %%
import numpy as np
import pandas as pd
import neurokit2 as nk
import re
import matplotlib.pyplot as plt

from blinkit import sp_filters, io, blink_params, data
from blinkit import plot as myplot


COL_FILTER = ["Onsets", "Offsets"]



def interpret_header(head: list[str]):
    """
    This function interprets the header and 
    generates a dictionary with all relevant info
    """
    ## Tag
    file_info = {}
    tag = ""
    for line in head:
        m_tag = re.search(r"\b.*\bblinks", line)

        if m_tag != None:
            temp = re.split('\s', m_tag.group(0))
            tag = temp[0].lower()
    file_info['tag'] = tag

    return file_info

class FileViewer:
    blinks_df = None
    eog_list = None

    cleaned_eog = None
    raw_eog = None
    file_path = ""
    tag = ""
    channel = 0
    info_dict = None

    def __init__(self, file_path: str, tag: str = None, channel: int = 1, apply_lowpass = True) -> None:
        """
        Read and add file to database
        """
        df_eog, head = io.read_Voltsxls(file_path, include_header=True)

        info_dict = interpret_header(head)
        channel_name = f"CH{channel}"
        assert channel_name in df_eog.columns, f"Channel {channel} not found"

        use_data = df_eog["CH1"].values
        self.raw_eog = df_eog
        if apply_lowpass:
            use_data = sp_filters.butter_lowpass_filter(use_data, 25, 1000, order = 2)
        self.cleaned_eog = df_eog.copy()
        for i in self.cleaned_eog.columns:
            temp = sp_filters.butter_lowpass_filter(self.cleaned_eog[i], 25, 1000, order = 2)
            self.cleaned_eog[i] = data.detrend_standardize(temp)
        eog_cleaned = data.detrend_standardize(use_data)
        blinks = nk.signal_findpeaks(eog_cleaned, relative_height_min=0)
        df_blinks = pd.DataFrame(blinks)

        blink_filter =  np.where((~pd.isna(df_blinks["Onsets"]) & (~pd.isna(df_blinks["Offsets"]))))[0]
        col_filter = ["Onsets", "Offsets"]
        all_blink_lims = df_blinks.loc[blink_filter, col_filter].values
        valid_filt = data.filter_blinks(eog_cleaned, all_blink_lims, threshold=0.2, duration = 900)
        valid_blinks = all_blink_lims[np.where(valid_filt)[0]]
        df_blinks = blink_params.gen_blinktable(eog_cleaned, valid_blinks)
        blink_params.blink_stats.perform(eog_cleaned, df_blinks)

        self.blinks_df = df_blinks
        self.channel = channel
        self.eog_list = eog_cleaned
        self.file_path = file_path
        self.info_dict = info_dict

        if tag == None:
            if info_dict['tag'] == "":
                self.tag = file_path
                print(f"Tag not specified, using '{file_path}'")
            else:
                temp = info_dict["tag"]
                print(f"Tag not specified, using '{temp}'")
                self.tag = temp
        else: self.tag = tag

        self.blinks_df['tag'] = np.tile(self.tag, len(self.blinks_df))
    
    def num_blinks(self):
        """
        Return the number of blinks in file
        """
        return len(self.blinks_df)
    
    def give_data(self):
        """
        Return the eog, blink table pair
        """
        return self.eog_list, self.blinks_df
    
    def give_blinks(self):
        """
        Return all blink lists
        """
        return [p[0] for p in self]

    def plot_eog(self, plot_range: list[int] = None, flag_blinks = True, title: str = None, ax = None, channel = 1):
        """
        Plot the EOG. Other specific parameters are available for plotting
        """
        channel_name = f"CH{channel}"
        blink_eog, df_blinks = self.cleaned_eog[channel_name], self.blinks_df
        #TODO: Do the ax=None matplotlib axis functionality
        
        plt.figure(figsize = (10, 8), facecolor = 'white')
        if plot_range == None: plot_range = [0, len(blink_eog)]
        use_range = [min(i, len(blink_eog)) for i in plot_range]
        plt.plot(blink_eog)
        
        if flag_blinks:
            visible_blinks = np.where((df_blinks["Onsets"] < use_range[1]) & (df_blinks["Offsets"] > use_range[0]))[0]
            blink_lims = df_blinks.loc[visible_blinks, BlinkFrame.col_filter].values
            for i in range(len(blink_lims)):
                #elems = np.arange(*blink_lims[i], dtype = int)
                #plt.plot(elems, eog_cleaned[elems], alpha = 0.5, color =  'crimson')
                box = myplot.create_rect(blink_eog, blink_lims[i], alpha = 0.3, fill = True)
                plt.gca().add_patch(box)
        plt.grid()
        plt.xlim(use_range[0], use_range[1])
        plt.title(title)
        plt.ylabel("EOG")
        plt.xlabel("Array")
    
    def plot_overlap(self, filter_func = lambda blink, tag: 1, group_func = lambda blink, tag: None, ax = None):
        """
        Overlap blinks by centering them at 0. Configurable filter functions and grouping functions
        """
        plt.figure(figsize=(10, 8))
        plt.grid()
        for blink_eog, blink in self:
            if filter_func(blink_eog, self.tag):
                peak_index = blink["Peaks"] - blink["Onsets"]
                x = np.arange(-peak_index, len(blink_eog)-peak_index)
                #TODO Add grouping functionality (color?)
                plt.plot(x, blink_eog)

    def __iter__(self):
        self.iter_num = 0
        return self
    
    def __next__(self):
        if self.iter_num < len(self.blinks_df):
            blink = self.blinks_df.iloc[self.iter_num]
            blink_lims = blink[COL_FILTER].values
            blink_eog = self.eog_list[int(blink_lims[0]): int(blink_lims[1])]
            
            self.iter_num +=1
            return blink_eog, blink.copy()
        else: raise StopIteration

class BlinkFrame:
    fviewers = None

    def __init__(self) -> None:
        """
        This is a general structure that holds and compiles the various blink files
        """
        self.fviewers = {}

    def add_file(self, file_path: str, tag: str = None, channel: int = 1, apply_lowpass = True) -> None:
        """
        Read and add file to database
        """
        fv = FileViewer(file_path, tag, channel, apply_lowpass)
        assert fv.tag not in self.fviewers.keys(), f"Tag {fv.tag} already exists"
        self.fviewers[fv.tag] = fv


    def find(self, tag: str) -> FileViewer:
        """
        Return the corresponding `blink_eog` and `blinks_df` 
        for the label.
        """
        if tag not in self.fviewers.keys(): return None
        fv = self.fviewers[tag]
        return fv
    
    def gen_blinks(self, tags: list[str] = None):
        """
        Aggregate and return blinks of specified files. If 
        `None` or `[]` no labels are specified, all available 
        blinks are generated.
        """
        file_blinks = []
        if tags == None:
            tags = self.fviewers.keys()

        for tag in tags:
            temp_fv = self.find(tag)
            file_blinks.append(temp_fv.give_blinks())
        return np.concatenate(file_blinks, dtype = object)
    
    def label_blinks(self, tags: list[str] = None, label_func = lambda blink, tag: tag):
        """
        Generate and label blinks
        """
        if tags == None:
            tags = self.fviewers.keys()

        all_labels = []
        all_blink_eogs = []
        all_df = []
        for tag in tags:
            temp_fv = self.find(tag)
            for blink_eog, blink in temp_fv:
                blink_label = label_func(blink_eog, tag)
                all_labels.append(blink_label)
                all_blink_eogs.append(blink_eog)
                all_df.append(blink)
        all_df = pd.DataFrame(all_df)
        all_df.reset_index(inplace=True)
        return all_blink_eogs, all_df, all_labels
# %%

# %%
