import sys

sys.path.insert(0, '../')

import numpy as np
# Datasets
from aif360.datasets import GermanDataset
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions \
    import get_distortion_adult, get_distortion_german, get_distortion_compas

np.random.seed(1)


def get_optim_options(dataset_used, protected_attribute_used):
    if dataset_used == "adult":
        optim_options = {
            "distortion_fun": get_distortion_adult,
            "epsilon": 0.05,
            "clist": [0.99, 1.99, 2.99],
            "dlist": [.1, 0.05, 0]
        }

    elif dataset_used == "german":
        if protected_attribute_used == 1:
            optim_options = {
                "distortion_fun": get_distortion_german,
                "epsilon": 0.05,
                "clist": [0.99, 1.99, 2.99],
                "dlist": [.1, 0.05, 0]
            }

        else:
            optim_options = {
                "distortion_fun": get_distortion_german,
                "epsilon": 0.1,
                "clist": [0.99, 1.99, 2.99],
                "dlist": [.1, 0.05, 0]
            }

    elif dataset_used == "compas":
        optim_options = {
            "distortion_fun": get_distortion_compas,
            "epsilon": 0.05,
            "clist": [0.99, 1.99, 2.99],
            "dlist": [.1, 0.05, 0]}

    return optim_options


def get_german_dataset():
    dataset = GermanDataset(protected_attribute_names=['sex'])
    dataset_name = 'german'
    dataset_orig_train, dataset_orig_val, dataset_orig_test = dataset.split([0.5, 0.8], shuffle=True)
    sens_ind = 0
    sens_attr = dataset_orig_train.protected_attribute_names[sens_ind]
    unprivileged_groups = [{sens_attr: v} for v in
                           dataset_orig_train.unprivileged_protected_attributes[sens_ind]]
    privileged_groups = [{sens_attr: v} for v in
                         dataset_orig_train.privileged_protected_attributes[sens_ind]]
    return dataset_orig_train, dataset_orig_val, dataset_orig_test, unprivileged_groups, privileged_groups
