# Sources:
# https://github.com/Trusted-AI/AIF360/blob/master/examples/tutorial_medical_expenditure.ipynb
# https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_meta_classifier.ipynb
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
from textwrap import wrap
import numpy as np
import pandas as pd
import json
from collections import defaultdict

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid

from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.datasets import BinaryLabelDataset
from aif360.explainers import MetricTextExplainer
from aif360.algorithms.preprocessing import (
    Reweighing, DisparateImpactRemover, LFR, OptimPreproc,)
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from typeguard import typechecked
from typing import Dict, List, Tuple, Union
from pathlib import Path
import os


import constants
from constants import DATASETS, CLASSIFIERS, FAIRBOOST_HYPERPARAMETERS
from FairBoost.main import FairBoost, Bootstrap_type
from FairBoost import wrappers


np.random.seed(0)


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
        print(f"\nevaluating FairBoost with classifier {clf_name}")
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
def initOptimPreproc(dataset_info: Dict, hyperparameters={}) -> wrappers.Preprocessing:
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
    OP = initOptimPreproc(
        dataset_info, hyperparameters['preprocessing']['OptimPreproc'])
    LFR_transformer = init_LFR(
        dataset_info, hyperparameters['preprocessing']['LFR'])
    pp = [RW, DIR, OP, LFR_transformer]

    for clf_name, clf in CLASSIFIERS.items():
        print(f"\nevaluating FairBoost with classifier {clf_name}")
        try:
            ens = FairBoost(clf, pp, **hyperparameters['init'])
            ens = ens.fit(train_dataset)
            y_pred = ens.predict(test_dataset)

            classified_dataset = test_dataset.copy()
            classified_dataset.labels = y_pred
            results[clf_name] = measure_results(
                test_dataset, classified_dataset, dataset_info)
        except Exception as e:
            print(f"Failed to run Fairboost with given hyper params. The error msg is:")
            print(e)

    return dict(results)


@typechecked
def measure_results(test_dataset: BinaryLabelDataset, classified_dataset: BinaryLabelDataset, dataset_info: Dict) -> Dict:
    """
    Computes fairness and accuracy metrics.

    :param test_dataset: an AIF360 dataset containing the test examples with their labels
    :param classified_dataset: an AIF360 dataset containing the test examples with the predicted labels
    :param dataset_info: information about the dataset including privileged and unprivileged groups
    :return: a dictionary of accuracy and fairness metrics
    """
    classification_metric = ClassificationMetric(
        dataset=test_dataset,
        classified_dataset=classified_dataset,
        unprivileged_groups=dataset_info["unprivileged_groups"],
        privileged_groups=dataset_info["privileged_groups"],
    )

    # calculate metrics
    accuracy = accuracy_score(test_dataset.labels, classified_dataset.labels)
    disparate_impact = classification_metric.disparate_impact()
    average_odds_difference = classification_metric.average_odds_difference()

    print(f"accuracy {accuracy}")
    print(f"disparate_impact {disparate_impact}")
    print(f"average odds difference {average_odds_difference}")

    return {
        "accuracy": accuracy,
        "disparate_impact": disparate_impact,
        "average_odds_difference": average_odds_difference,
    }


def main():
    results = defaultdict(dict)

    for dataset_name, dataset_info in DATASETS.items():

        print(f"\n\n$$$$in dataset {dataset_name}$$$$$\n")

        dataset: BinaryLabelDataset = dataset_info["original_dataset"]

        train_split, test_split = dataset.split([0.7], shuffle=True, seed=0)
        # train_split, test_split = initial_preprocessing(train_split, test_split)

        results[dataset_name]["baseline"] = []
        for hyperparameters in ParameterGrid(constants.FairBoost_param_grid):
            performance_metrics = train_test_bagging_baseline(
                train_split, test_split, dataset_info, hyperparameters
            )
            results[dataset_name]["baseline"].append(
                {"hyperparameters": hyperparameters,
                    "results": performance_metrics}
            )

        results[dataset_name]["fairboost"] = []
        for hyperparameters in ParameterGrid(FAIRBOOST_HYPERPARAMETERS):

            performance_metrics = train_test_fairboost(
                train_split, test_split, dataset_info, hyperparameters)

            # record the used hyperparameters
            results[dataset_name]["fairboost"].append(
                {"hyperparameters": hyperparameters,
                    "results": performance_metrics}
            )

    # save the results to file
    save_results(filename='fairboost', results=results)


@typechecked
def save_results(filename: str, results: Dict):
    """
    Saves a given dictionary to a JSON file.

    :param filename: The name of the output file
    :param results: The results to be saved
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    results_dir = Path(dir_path, "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(f"{results_dir}/{filename}.json", "w") as fp:
        json.dump(results, fp, indent=4)


if __name__ == "__main__":
    main()
