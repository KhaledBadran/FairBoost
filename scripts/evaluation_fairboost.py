# Sources:
# https://github.com/Trusted-AI/AIF360/blob/master/examples/tutorial_medical_expenditure.ipynb
# https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_meta_classifier.ipynb
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
from textwrap import wrap
from unittest import result
import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid

from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import BinaryLabelDataset
from aif360.explainers import MetricTextExplainer
from aif360.algorithms.preprocessing import (
    Reweighing, DisparateImpactRemover, LFR, OptimPreproc,)
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from typeguard import typechecked
from typing import Dict


from constants.splits import DATASETS, CLASSIFIERS, FAIRBOOST_HYPERPARAMETERS, FairBoost_param_grid, SEEDS
from FairBoost.main import FairBoost, Bootstrap_type
from FairBoost import wrappers
from utils import save_results, measure_results, merge_results_array

np.random.seed(42)


@typechecked
def train_test_bagging_baseline(train_dataset: BinaryLabelDataset,
                                test_dataset: BinaryLabelDataset,
                                dataset_info: Dict,
                                hyperparameters: Dict) -> Dict:
    """
    Trains Fairboost for the given hyperparameters, with different classifiers, using no preprocessing functions. 
    Then computes fairness and accuracy metrics.

    :param train_dataset: an AIF360 dataset containing the training examples with their labels
    :param test_dataset: an AIF360 dataset containing the test examples with their labels
    :param dataset_info: information about the dataset including privileged and unprivileged groups
    :param hyperparameters: hyperparameters to initialize fairboost
    :return: a dictionary of accuracy and fairness metrics (e.g., disparate_impact) with the values calculated using
    a trained model
    """
    results = defaultdict(dict)
    pp = [wrappers.NoPreprocessing() for _ in range(4)]
    for clf_name, clf in CLASSIFIERS.items():
        print(f"\nFairboost classifier name: {clf_name}")
        ens = FairBoost(clf, pp, **hyperparameters)
        ens = ens.fit(train_dataset)
        y_pred = ens.predict(test_dataset)

        classified_dataset = test_dataset.copy()
        classified_dataset.labels = y_pred
        results[clf_name] = measure_results(
            test_dataset, classified_dataset, dataset_info)
    return results


@typechecked
def init_reweighting(dataset_info: Dict, hyperparameters={}) -> wrappers.Preprocessing:
    """
    Initializes the reweighting algorithm so it can be used by Fairboost.

    :param dataset_info: information about the dataset including privileged and unprivileged groups
    :param hyperparameters: hyperparameters to initialize the reweighing algorithm
    :return: The reweighing preprocessing function to be used by Fairboost.
    """
    RW = Reweighing(
        privileged_groups=dataset_info["privileged_groups"],
        unprivileged_groups=dataset_info["unprivileged_groups"],
    )
    return wrappers.Reweighing(RW)


@typechecked
def init_DIR(dataset_info: Dict, hyperparameters={}) -> wrappers.Preprocessing:
    """
    Initializes the DIR algorithm so it can be used by Fairboost.

    :param dataset_info: information about the dataset including privileged and unprivileged groups
    :param hyperparameters: hyperparameters to initialize the DIR algorithm
    :return: The DIR preprocessing function to be used by Fairboost.
    """
    DIR = DisparateImpactRemover(
        sensitive_attribute=dataset_info["sensitive_attribute"],
        repair_level=hyperparameters['init']["repair_level"],
    )
    return wrappers.DIR(DIR)


@typechecked
def init_OptimPreproc(dataset_info: Dict, hyperparameters={}) -> wrappers.Preprocessing:
    """
    Initializes the OptimPreproc algorithm so it can be used by Fairboost.

    :param dataset_info: information about the dataset including privileged and unprivileged groups
    :param hyperparameters: hyperparameters to initialize the OptimPreproc algorithm
    :return: The OptimPreproc preprocessing function to be used by Fairboost.
    """
    OP = OptimPreproc(OptTools, dataset_info["optim_options"], verbose=False)
    return wrappers.OptimPreproc(OP)


@typechecked
def init_LFR(dataset_info: Dict, hyperparameters={}) -> wrappers.Preprocessing:
    """
    Initializes the LFR algorithm so it can be used by Fairboost.

    :param dataset_info: information about the dataset including privileged and unprivileged groups
    :param hyperparameters: hyperparameters to initialize the LFR algorithm
    :return: The LFR preprocessing function to be used by Fairboost.
    """
    LFR_transformer = LFR(
        unprivileged_groups=dataset_info["unprivileged_groups"],
        privileged_groups=dataset_info["privileged_groups"],
        k=hyperparameters['init']["k"],
        Ax=hyperparameters['init']["Ax"],
        Ay=hyperparameters['init']["Ay"],
        Az=hyperparameters['init']["Az"],
        verbose=0,  # Default parameters
    )
    return wrappers.LFR(LFR_transformer, transform_params=hyperparameters['transform'])


@typechecked
def train_test_fairboost(train_dataset: BinaryLabelDataset,
                         test_dataset: BinaryLabelDataset,
                         dataset_info: Dict,
                         hyperparameters: Dict) -> Dict:
    """
    Trains Fairboost for the given hyperparameters, with different classifiers. 
    Then computes fairness and accuracy metrics.

    :param train_dataset: an AIF360 dataset containing the training examples with their labels
    :param test_dataset: an AIF360 dataset containing the test examples with their labels
    :param dataset_info: information about the dataset including privileged and unprivileged groups
    :param hyperparameters: hyperparameters to initialize fairboost
    :return: a dictionary of accuracy and fairness metrics
    """
    results = defaultdict(dict)
    RW = init_reweighting(
        dataset_info, hyperparameters['preprocessing']['Reweighing'])
    DIR = init_DIR(
        dataset_info, hyperparameters['preprocessing']['DisparateImpactRemover'])
    OP = init_OptimPreproc(
        dataset_info, hyperparameters['preprocessing']['OptimPreproc'])
    LFR_transformer = init_LFR(
        dataset_info, hyperparameters['preprocessing']['LFR'])
    pp = [RW, DIR, OP, LFR_transformer]

    for clf_name, clf in CLASSIFIERS.items():
        print(f"\nFairboost classifier name: {clf_name}")
        try:
            # Training + prediction
            ens = FairBoost(clf, pp, **hyperparameters['init'])
            ens = ens.fit(train_dataset)
            y_pred = ens.predict(test_dataset)

            # Measuring metrics
            classified_dataset = test_dataset.copy()
            classified_dataset.labels = y_pred
            results[clf_name] = measure_results(
                test_dataset, classified_dataset, dataset_info)
        except Exception as e:
            print(f"Failed to run Fairboost with given hyper params. The error msg is:")
            print(e)

    return dict(results)


@typechecked
def evaluate_baseline(results: defaultdict, dataset: BinaryLabelDataset, dataset_name: str, dataset_info: dict) -> defaultdict:
    """
    Run Fairboost with no preprocessing using different hyperparameter configurations.
    Measure and save the performances.

    :param results: The dictionnary storing results for the run
    :param dataset: an AIF360 dataset containing the test examples with their labels
    :param dataset_name: The name of the dataset
    :param dataset_info: information about the dataset including privileged and unprivileged groups
    :return: The updated results dictionnary
    """
    results[dataset_name]["baseline"] = []

    # Measuring Fairboost performances for different hyperparameter configurations (without unfairness mitigation techniques)
    for hyperparameters in ParameterGrid(FairBoost_param_grid):
        results[dataset_name]["baseline"].append(
            {"hyperparameters": hyperparameters,
                "results": []}
        )
        # Splitting dataset over different seeds
        for seed in SEEDS:
            train_split, test_split = dataset.split(
                [0.7], shuffle=True, seed=seed)
            # Measuring model performance
            performance_metrics = train_test_bagging_baseline(
                train_split, test_split, dataset_info, hyperparameters
            )
            results[dataset_name]["baseline"][-1]['results'].append(
                performance_metrics)

        # Merging results for clarity
        results[dataset_name]["baseline"][-1]['results'] = merge_results_array(
            results[dataset_name]["baseline"][-1]['results'])
    return results


@typechecked
def evaluate_fairboost(results: defaultdict, dataset: BinaryLabelDataset, dataset_name: str, dataset_info: dict):
    """
    Run Fairboost using different hyperparameter configurations.
    Measure and save the performances.

    :param results: The dictionnary storing results for the run
    :param dataset: an AIF360 dataset containing the test examples with their labels
    :param dataset_name: The name of the dataset
    :param dataset_info: information about the dataset including privileged and unprivileged groups
    :return: The updated results dictionnary
    """
    results[dataset_name]["fairboost"] = []
    n_combinations = len(list(ParameterGrid(FAIRBOOST_HYPERPARAMETERS)))

    # Measuring Fairboost performances for different hyperparameter configurations
    for i, hyperparameters in enumerate(ParameterGrid(FAIRBOOST_HYPERPARAMETERS)):
        print(f"\n---------- Progress: {i}/{n_combinations} ----------")
        results[dataset_name]["fairboost"].append(
            {"hyperparameters": hyperparameters,
                "results": []}
        )

        # Splitting dataset over different seeds
        for seed in SEEDS:
            train_split, test_split = dataset.split(
                [0.7], shuffle=True, seed=seed)
            # Measuring model performance
            performance_metrics = train_test_fairboost(
                train_split, test_split, dataset_info, hyperparameters)
            results[dataset_name]["fairboost"][-1]['results'].append(
                performance_metrics)

        # Merging results for clarity
        results[dataset_name]["fairboost"][-1]['results'] = merge_results_array(
            results[dataset_name]["fairboost"][-1]['results'])
    return results


def main():
    results = defaultdict(dict)

    for dataset_name, dataset_info in DATASETS.items():

        print(f"\n\n$$$$in dataset {dataset_name}$$$$$\n")

        dataset: BinaryLabelDataset = dataset_info["original_dataset"]

        print(f"\n\n---------- Baselines ----------")
        results = evaluate_baseline(
            results, dataset, dataset_name, dataset_info)

        print(f"\n\n---------- Fairboost ----------")
        results = evaluate_fairboost(
            results, dataset, dataset_name, dataset_info)

    # save the results to file
    save_results(filename='fairboost_splits', results=results)


if __name__ == "__main__":
    main()
