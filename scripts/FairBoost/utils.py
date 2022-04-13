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