import os
from os import scandir as sdir_

import pandas as pd


def df_sdir(dataroot: str, cols=['fpath', 'fname', 'slide_id']):
    """
    Returns pd.DataFrame of the file paths and fnames of contents in dataroot.

    Args:
        dataroot (str): path to files.

    Returns:
        (pandas.Dataframe): pd.DataFrame of the file paths and fnames of contents in dataroot (make default cols: ['fpath', 'fname_ext', 'fname_noext']?)
    """
    return pd.DataFrame([(e.path, e.name, os.path.splitext(e.name)[0]) for e in sdir_(dataroot)], columns=cols)

# TODO: Fix doc + also make function for ldir_diff
def series_diff(s1, s2, dtype='O'):
    r"""
    Returns set difference of two pd.Series.
    """
    return pd.Series(list(set(s1).difference(set(s2))), dtype=dtype)

series_difference = series_diff