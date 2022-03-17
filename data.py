import sys

sys.path.insert(0, '../')

import numpy as np
# Datasets
from aif360.datasets import GermanDataset

np.random.seed(1)


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
