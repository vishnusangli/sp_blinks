import pandas as pd
import re
import numpy as np

match = np.vectorize(re.match)
def read_Voltsxls(file_path: str, readings_header: str = 'CH1\tCH2\tCH3\t', include_header = False):
    """
    Personalized funtion to read and interpret our eog data files. \n

    full path to the files must be listed
    """
    f = open(file_path, 'r')
    file_aslist = [line.rstrip('\n') for line in f]
    table_start = np.where(match('CH\d', file_aslist) != None)[0][0]
    df_eog = pd.read_csv(file_path, sep ="\t", skiprows = table_start - 1)
    for i in df_eog.columns:
        if 'CH' not in i:
            df_eog = df_eog.drop(i, axis = 1)
    df_eog.set_axis(['CH1', 'CH2', 'CH3'], axis = 1)
    if include_header: return df_eog, file_aslist[:table_start]
    return df_eog
