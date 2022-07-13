import pandas as pd


def read_Voltsxls(file_path: str, readings_header: str = 'CH1\tCH2\tCH3\t'):
    """
    Personalized funtion to read and interpret our eog data files. \n

    full path to the files must be listed
    """
    f = open(file_path, 'r')
    file_aslist = [line.rstrip('\n') for line in f]
    table_start = file_aslist.index(readings_header)
    df_eog = pd.read_csv(file_path, sep ="\t", skiprows = table_start -1)
    df_eog = df_eog.drop('Unnamed: 3', axis = 1)
    return df_eog
