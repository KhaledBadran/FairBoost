from typing import List, Tuple
import warnings
import os
import sys
import numpy as np


def quiet(func, args):
    '''
    Runs a given function with the given args while muting logs and warnings.

            Parameters:
                    func: The function to run
                    args: A list of positional arguments for the called function

            Returns:
                    pp_data (list): List with the different preprocessed data sets
    '''
    with warnings.catch_warnings():
        sys.stdout = open(os.devnull, 'w')
        warnings.simplefilter("ignore")
        res = func(*args)
        sys.stdout = sys.__stdout__
    return res


def concat_datasets(d1, d2):
    """
    Concats two datasets d1 and d2 where d1 may be empty.
    """
    if len(d1) > 0:
        return np.concatenate((d1, d2))
    else:
        return d2


def merge_tuples(datasets: List[Tuple]) -> np.array:
    '''
    Merges a tuple of n rows and 3 elements into a np array of n rows.
            Parameters:
                    datasets: List with tuples to be merged.

            Returns:
                    res: the concatenated np array
    '''
    res = []
    for dataset in datasets:
        X, y, w = dataset[0], dataset[1], np.expand_dims(
            dataset[2], axis=-1)
        m = np.concatenate([X, y, w], axis=-1)
        res.append(m)
    return np.array(res)


def unmerge_tuples(datasets: List[np.array]) -> List[Tuple]:
    '''
    Unmerges a np array of n rows into a tuple of n rows and 3 elements.
            Parameters:
                    datasets: A list of numpy arrays to unmerge

            Returns:
                    res: A list of tuples.
    '''
    res = []
    for dataset in datasets:
        res.append((dataset[:, :-2], dataset[:, -2], dataset[:, -1]))
    return res
