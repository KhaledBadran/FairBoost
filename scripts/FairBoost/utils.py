import warnings
import os
import sys


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
