from typing import List, Tuple
import warnings
import os
import sys
import numpy as np
import scipy.spatial.distance as dist


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


def get_avg_dist_arr(datasets: np.array, dist_func=dist.cosine) -> np.array:
    '''
    Measure the average distance between rows of numpy arrays. 
    It is repeated for each datasets. 

            Parameters:
                    datasets: a numpy array of datasets

            Returns:
                    dist_arr (np.array): Array with the distance for each instance of each "cleaned" data set.
    '''
    # Remove weights from dataset
    datasets = datasets[:, :, :-1]
    # Swap the first two dimensions so we iterate over instances instead of data sets
    datasets = datasets.transpose([1, 0, 2])
    # Initializing the average distances array
    dist_arr = np.zeros(shape=(datasets.shape[:-1]))
    # Fill the avg distances array
    for i, pp_instances in enumerate(datasets):
        for j, pp_instance_j in enumerate(pp_instances):
            distances = []
            for k, pp_instance_k in enumerate(pp_instances):
                d = dist_func(pp_instance_j, pp_instance_k)
                d = np.abs(d)
                distances.append(d)
            # One entry is zero (the distance with itself). Do not consider it in the mean.
            dist_arr[i, j] = np.sum(distances)/(len(pp_instances)-1)

    dist_arr = dist_arr.transpose([1, 0])
    return dist_arr
