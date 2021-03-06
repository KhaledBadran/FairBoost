# optim_option values is from https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_optim_data_preproc.ipynb
from typing import Dict, List
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import (
    load_preproc_data_adult,
    load_preproc_data_german,
    load_preproc_data_compas,
)
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions import (
    get_distortion_adult,
    get_distortion_german,
    get_distortion_compas,
)
from typeguard import typechecked
from sklearn.model_selection import ParameterGrid
from .enums import Preproc_name, Dataset_name


@typechecked
def initialize_german_dataset(protected_attributes: List = ["sex"]) -> BinaryLabelDataset:
    """
    Initializes the german dataset

    :param protected_attributes: The protected attributes of the dataset
    :return: The german dataset
    """
    ds = load_preproc_data_german(protected_attributes)
    ds.labels = ds.labels % 2  # turns 2s into 0 while keeping 1s the same
    ds.favorable_label = 1
    ds.unfavorable_label = 0

    # set the correct labels because the labels changed and its important for Optimized Preprocessing Distortion
    ds.label_map = {1.0: 'Good Credit', 0.0: 'Bad Credit'}
    ds.metadata['label_maps'] = [{1.0: 'Good Credit', 0.0: 'Bad Credit'}]

    return ds


@typechecked
def initialize_adult_dataset(protected_attributes: List = ["sex"]) -> BinaryLabelDataset:
    """
    Initializes the adult dataset

    :param protected_attributes: The protected attributes of the dataset
    :return: The adult dataset
    """
    return load_preproc_data_adult(protected_attributes)


@typechecked
def initialize_compass_dataset(protected_attributes: List = ["sex"]) -> BinaryLabelDataset:
    """
    Initializes the compass dataset

    :param protected_attributes: The protected attributes of the dataset
    :return: The compass dataset
    """

    ds = load_preproc_data_compas(protected_attributes)
    ds.labels = 1 - ds.labels  # turns 1s into 0s and 0s to 1s
    ds.favorable_label = 1
    ds.unfavorable_label = 0

    # set the correct labels because the labels changed and its important for Optimized Preprocessing Distortion
    ds.label_map = {0.0: 'Did recid.', 1.0: 'No recid.'}
    ds.metadata['label_maps'] = [{0.0: 'Did recid.', 1.0: 'No recid.'}]

    return ds


@typechecked
def get_preproc_hyperparameters(dataset_name: Dataset_name) -> Dict:
    """
    Returns the hyperparameters of all the preprocessing functions
    we have selected for a specific dataset.

    :param dataset_name: The name of the dataset 
    :return: The hyperparameters of every preprocessing function
    """
    if dataset_name == Dataset_name.GERMAN:
        return {
            Preproc_name.OptimPreproc: [{
                "optim_options": {
                    "distortion_fun": get_distortion_german,
                    "epsilon": 0.1,
                    "clist": [0.99, 1.99, 2.99],
                    "dlist": [0.1, 0.05, 0],
                },
            }],
            Preproc_name.LFR: [{'init': {'Ax': 0.1, 'Ay': 1.0, 'Az': 0, 'k': 5}, 'transform': {'threshold': 0.5}}],
            Preproc_name.DisparateImpactRemover: [{"init": {"repair_level": 0.5}}],
            Preproc_name.Reweighing: [{}]
        }
    elif dataset_name == Dataset_name.ADULT:
        return {
            Preproc_name.OptimPreproc: [{
                "optim_options": {
                    "distortion_fun": get_distortion_adult,
                    "epsilon": 0.05,
                    "clist": [0.99, 1.99, 2.99],
                    "dlist": [0.1, 0.05, 0],
                },
            }],
            Preproc_name.LFR: [{'init': {'Ax': 0.1, 'Ay': 1.0, 'Az': 1.0, 'k': 5}, 'transform': {'threshold': 0.5}}],
            Preproc_name.DisparateImpactRemover: [{"init": {"repair_level": 0.5}}],
            Preproc_name.Reweighing: [{}]
        }
    elif dataset_name == Dataset_name.COMPASS:
        return {
            Preproc_name.OptimPreproc: [{
                "optim_options": {
                    "distortion_fun": get_distortion_compas,
                    "epsilon": 0.05,
                    "clist": [0.99, 1.99, 2.99],
                    "dlist": [0.1, 0.05, 0],
                },
            }],
            Preproc_name.LFR: [{'init': {'Ax': 0.01, 'Ay': 0.1, 'Az': 1.0, 'k': 5}, 'transform': {'threshold': 0.5}}],
            Preproc_name.DisparateImpactRemover: [{"init": {"repair_level": 0.5}}],
            Preproc_name.Reweighing: [{}]
        }
    else:
        raise Exception('Error in dataset hyperparameter initialization')


@typechecked
def get_LFR_hyperparameters_search(dataset_name: Dataset_name):
    """
    Returns the different hyperparameters we searched for each dataset,
    for each unfairness mitigation technique.
    It must replace "get_preproc_hyperparameters" in constant file to be used. 

    :param dataset_name: The name of the dataset 
    :return: The hyperparameters of every preprocessing function
    """
    if dataset_name == Dataset_name.GERMAN:
        return {
            Preproc_name.LFR: list(ParameterGrid({
                "init": list(ParameterGrid({"Ax": [0.01, 0.1], "Ay": [0.1, 1.0, 10.0], "Az": [0, 0.1, 1.0, ], "k": [5, 10]})),
                "transform": [{"threshold": 0.5}],
            })),
        }
    elif dataset_name == Dataset_name.ADULT:
        return {
            Preproc_name.LFR: list(ParameterGrid({
                "init": list(ParameterGrid({"Ax": [0.01, 0.1], "Ay": [0.1, 1.0, 10.0], "Az": [0, 0.1, 1.0, ], "k": [5, 10]})),
                "transform": [{"threshold": 0.5}],
            })),
        }
    elif dataset_name == Dataset_name.COMPASS:
        return {
            Preproc_name.LFR: list(ParameterGrid({
                "init": list(ParameterGrid({"Ax": [0.01, 0.1], "Ay": [0.1, 1.0, 10.0], "Az": [0, 0.1, 1.0, ], "k": [5, 10]})),
                "transform": [{"threshold": 0.5}],
            })),
        }
    else:
        raise Exception('Error in dataset hyperparameter initialization')
