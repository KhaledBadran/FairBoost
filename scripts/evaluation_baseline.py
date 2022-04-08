# Sources:
# https://github.com/Trusted-AI/AIF360/blob/master/examples/tutorial_medical_expenditure.ipynb
# https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_meta_classifier.ipynb
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
import numpy as np
import pandas as pd
import json
from collections import defaultdict
from datetime import datetime

# Fairness metrics
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.datasets import BinaryLabelDataset
from sklearn.metrics import accuracy_score

# Explainers
from aif360.explainers import MetricTextExplainer

# Scalers
from sklearn.preprocessing import StandardScaler

# Bias mitigation techniques
from aif360.algorithms.preprocessing import (
    Reweighing,
    DisparateImpactRemover,
    LFR,
    OptimPreproc,
)

from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools

# Experiment constants
from constants.splits import (
    DATASETS,
    CLASSIFIERS,
    HYPERPARAMETERS,
    SEEDS,
    CLASSIFIERS_HYPERPARAMETERS,
)

from utils import save_results, measure_results, merge_results_array


# typechecking
from typeguard import typechecked
from typing import Dict, List, Tuple, Union

np.random.seed(0)


@typechecked
def train_test_models(
    train_dataset: BinaryLabelDataset,
    test_dataset: BinaryLabelDataset,
    dataset_info: Dict,
) -> Dict:
    """
    returns the accuracy and fairness metrics based after training and testing a model.

    :param train_dataset: an AIF360 dataset containing the training examples with their labels
    :param test_dataset: an AIF360 dataset containing the test examples with their labels
    :param dataset_info: information about the dataset including privileged and unprivileged groups
    :return: a dictionary of accuracy and fairness metrics (e.g., disparate_impact) with the values calculated using
    a trained model
    """
    results = defaultdict(dict)

    # Train models
    X_train, y_train = train_dataset.features, train_dataset.labels.ravel()
    X_test = test_dataset.features

    for clf_name, clf in CLASSIFIERS.items():
        print(f"\nevaluating classifier {clf_name}")

        # Check if the training labels are not all the same
        if len(set(y_train)) != 1:
            # Training + prediction
            clf.fit(X_train, y_train, sample_weight=train_dataset.instance_weights)
            y_pred = clf.predict(X_test)

            # Measuring metrics
            classified_dataset = test_dataset.copy()
            classified_dataset.labels = y_pred
            results[clf_name] = measure_results(
                test_dataset, classified_dataset, dataset_info
            )

        else:
            print(f"all training labels are same, classifier will not be trained")
            results[clf_name] = {}

    return dict(results)


@typechecked
def apply_reweighing(
    train_dataset: BinaryLabelDataset,
    test_dataset: BinaryLabelDataset,
    dataset_info: Dict,
) -> Tuple[BinaryLabelDataset, BinaryLabelDataset]:
    """
    :param train_dataset: an AIF360 dataset containing the training examples with their labels
    :param test_dataset: an AIF360 dataset containing the test examples with their labels
    :param dataset_info: information about the dataset including privileged and unprivileged groups
    :return: a train and test datasets that have been transformed via reweighing
    """
    # Apply standard scaler
    scaler = StandardScaler()
    scaler = scaler.fit(train_dataset.features)
    train_dataset.features = scaler.transform(train_dataset.features)
    test_dataset.features = scaler.transform(test_dataset.features)

    RW = Reweighing(
        privileged_groups=dataset_info["privileged_groups"],
        unprivileged_groups=dataset_info["unprivileged_groups"],
    )
    train_dataset_RW = RW.fit_transform(train_dataset)
    test_dataset_RW = test_dataset.copy(deepcopy=True)

    return train_dataset_RW, test_dataset_RW


def apply_DIR(
    train_dataset: BinaryLabelDataset,
    test_dataset: BinaryLabelDataset,
    dataset_info: Dict,
    hyperparameters: Dict,
) -> Tuple[BinaryLabelDataset, BinaryLabelDataset]:
    """
    :param train_dataset: an AIF360 dataset containing the training examples with their labels
    :param test_dataset: an AIF360 dataset containing the test examples with their labels
    :param dataset_info: information about the dataset including privileged and unprivileged groups
    :param hyperparameters: a dictionary containing the value of the repair_level parameter
    :return: a train and test datasets that have been transformed via Disparate Impact Remover technique
    """
    DIR = DisparateImpactRemover(
        sensitive_attribute=dataset_info["sensitive_attribute"],
        repair_level=hyperparameters["init"]["repair_level"],
    )

    train_dataset_DIR = DIR.fit_transform(train_dataset)
    test_dataset_DIR = DIR.fit_transform(test_dataset)

    return train_dataset_DIR, test_dataset_DIR


@typechecked()
def apply_OptimPreproc(
    train_dataset: BinaryLabelDataset,
    test_dataset: BinaryLabelDataset,
    dataset_info: Dict,
    hyperparameters: Dict,
) -> Tuple[BinaryLabelDataset, BinaryLabelDataset]:
    """
    :param train_dataset: an AIF360 dataset containing the training examples with their labels
    :param test_dataset: an AIF360 dataset containing the test examples with their labels
    :param dataset_info: information about the dataset including privileged and unprivileged groups
    :return: a train and test datasets that have been transformed via the Optimized Preprocessing technique
    """
    train_dataset_OP, test_dataset_OP = train_dataset.copy(
        deepcopy=True), test_dataset.copy(deepcopy=True)

    OP = OptimPreproc(
        OptTools, hyperparameters["optim_options"], verbose=False)

    OP = OP.fit(train_dataset)

    train_dataset_OP = OP.transform(train_dataset, transform_Y=True)
    train_dataset_OP = train_dataset.align_datasets(train_dataset_OP)

    test_dataset_OP = OP.transform(test_dataset, transform_Y=True)
    test_dataset_OP = test_dataset.align_datasets(test_dataset_OP)

    return train_dataset_OP, test_dataset_OP


@typechecked
def apply_LFR(
    train_dataset: BinaryLabelDataset,
    test_dataset: BinaryLabelDataset,
    dataset_info: Dict,
    hyperparameters: Dict,
) -> Tuple[BinaryLabelDataset, BinaryLabelDataset]:
    """
    :param train_dataset: an AIF360 dataset containing the training examples with their labels
    :param test_dataset: an AIF360 dataset containing the test examples with their labels
    :param dataset_info: information about the dataset including privileged and unprivileged groups
    :param hyperparameters: a dictionary containing both the hyperparameters for the LFR object (e.g., Ax) and the
    threshold for the transform method
    :return: a train and test datasets that have been transformed via the LFR technique
    """
    # Apply standard scaler
    scaler = StandardScaler()
    scaler = scaler.fit(train_dataset.features)
    train_dataset.features = scaler.transform(train_dataset.features)
    test_dataset.features = scaler.transform(test_dataset.features)

    LFR_transformer = LFR(
        unprivileged_groups=dataset_info["unprivileged_groups"],
        privileged_groups=dataset_info["privileged_groups"],
        k=hyperparameters["init"]["k"],
        Ax=hyperparameters["init"]["Ax"],
        Ay=hyperparameters["init"]["Ay"],
        Az=hyperparameters["init"]["Az"],
        verbose=0,  # Default parameters
    )

    LFR_transformer = LFR_transformer.fit(
        train_dataset, maxiter=5000, maxfun=5000)

    # Transform training data and align features
    train_dataset_LFR = LFR_transformer.transform(
        train_dataset, threshold=hyperparameters["transform"]["threshold"]
    )
    test_dataset_LFR = LFR_transformer.transform(
        test_dataset, threshold=hyperparameters["transform"]["threshold"]
    )

    return train_dataset_LFR, test_dataset_LFR


@typechecked
def apply_preprocessing_algo(
    algo_name: str,
    hyperparameters: Dict,
    train_dataset: BinaryLabelDataset,
    test_dataset: BinaryLabelDataset,
    dataset_info: Dict,
) -> Tuple[BinaryLabelDataset, BinaryLabelDataset]:
    """
    :param algo_name: name of the preprocessing technique to apply
    :param hyperparameters: used to tune the preprocessing technique (can be an empty dict)
    :param train_dataset: an AIF360 dataset containing the training examples with their labels
    :param test_dataset: an AIF360 dataset containing the test examples with their labels
    :param dataset_info: information about the dataset including privileged and unprivileged groups
    :return: a train and test datasets that have been transformed via one of the preprocessing techniques
    """

    # just an initialization
    train_dataset_transformed, test_dataset_transformed = train_dataset.copy(
        deepcopy=True), test_dataset.copy(deepcopy=True)

    try:
        if algo_name == "Reweighing":
            train_dataset_transformed, test_dataset_transformed = apply_reweighing(
                train_dataset, test_dataset, dataset_info
            )
        elif algo_name == "DisparateImpactRemover":
            train_dataset_transformed, test_dataset_transformed = apply_DIR(
                train_dataset, test_dataset, dataset_info, hyperparameters
            )

        elif algo_name == "OptimPreproc":
            train_dataset_transformed, test_dataset_transformed = apply_OptimPreproc(
                train_dataset, test_dataset, dataset_info, hyperparameters
            )

        elif algo_name == "LFR":
            train_dataset_transformed, test_dataset_transformed = apply_LFR(
                train_dataset, test_dataset, dataset_info, hyperparameters
            )

    except Exception as e:
        print(f"Failed to pre-process the dataset. The error msg is:")
        print(e)

    # Make sure test labels are not modified
    test_dataset_transformed.labels = test_dataset.labels

    return (
        train_dataset_transformed,
        test_dataset_transformed,
    )


@typechecked
def evaluate_baseline(
    results: defaultdict,
    dataset: BinaryLabelDataset,
    dataset_name: str,
    dataset_info: dict,
):
    """
    Run models using no unfairness mitigation techniques.
    Measure and save the performances.

    :param results: The dictionnary storing results for the run
    :param dataset: an AIF360 dataset containing the test examples with their labels
    :param dataset_name: The name of the dataset
    :param dataset_info: information about the dataset including privileged and unprivileged groups
    :return: The updated results dictionnary
    """
    results[dataset_name]["baseline"] = []

    # Splitting dataset over different seeds
    for seed in SEEDS:
        train_split, test_split = dataset.split([0.7], shuffle=True, seed=seed)
        train_dataset, test_dataset = train_split.copy(
            deepcopy=True), test_split.copy(deepcopy=True)

        # Apply standard scaler
        scaler = StandardScaler()
        scaler = scaler.fit(train_dataset.features)
        train_dataset.features = scaler.transform(train_dataset.features)
        test_dataset.features = scaler.transform(test_dataset.features)

        # Measuring model performance
        metrics = train_test_models(
            train_dataset, test_dataset, dataset_info=dataset_info)

        results[dataset_name]["baseline"].append(metrics)

    # Merging results for clarity
    results[dataset_name]["baseline"] = merge_results_array(
        results[dataset_name]["baseline"]
    )
    return results


@typechecked
def evaluate_mitigation_techniques(
    results: defaultdict,
    dataset: BinaryLabelDataset,
    dataset_name: str,
    dataset_info: dict,
):
    """
    Run the model using unfairness mitigation techniques while doing hyperparameter tuning.
    Measure and save the performances.

    :param results: The dictionnary storing results for the run
    :param dataset: an AIF360 dataset containing the test examples with their labels
    :param dataset_name: The name of the dataset
    :param dataset_info: information about the dataset including privileged and unprivileged groups
    :return: The updated results dictionnary
    """
    # Per-dataset hyperparameters overwrite general hyperparameters
    m_hyperparameters = {**HYPERPARAMETERS, **dataset_info['hyperparams']}
    for (
        debaiasing_algo_name,
        hyperparameters_space,
    ) in m_hyperparameters.items():
        print(f"\n\n####After applying {debaiasing_algo_name}######\n")

        results[dataset_name][debaiasing_algo_name] = []

        for hyperparameters in hyperparameters_space:
            results[dataset_name][debaiasing_algo_name].append(
                {"hyperparameters": hyperparameters, "results": []}
            )

            # Splitting dataset over different seeds
            for seed in SEEDS:
                train_split, test_split = dataset.split(
                    [0.7], shuffle=True, seed=seed)
                # Transforming datasets with unfairness mitigation technique
                (
                    train_split_transformed,
                    test_split_transformed,
                ) = apply_preprocessing_algo(
                    debaiasing_algo_name,
                    hyperparameters,
                    train_split.copy(deepcopy=True),
                    test_split.copy(deepcopy=True),
                    dataset_info,
                )
                # Measuring model performance
                performance_metrics = train_test_models(
                    train_split_transformed,
                    test_split_transformed,
                    dataset_info=dataset_info,
                )
                results[dataset_name][debaiasing_algo_name][-1]["results"].append(
                    performance_metrics
                )

            # Merging results for clarity
            results[dataset_name][debaiasing_algo_name][-1][
                "results"
            ] = merge_results_array(
                results[dataset_name][debaiasing_algo_name][-1]["results"]
            )
    return results


def main():
    results = defaultdict(dict)

    for dataset_name, dataset_info in DATASETS.items():

        print(f"\n\n$$$$in dataset {dataset_name}$$$$$\n")

        dataset: BinaryLabelDataset = dataset_info["original_dataset"]

        print(f"\n\n---------- Baselines ----------")
        results = evaluate_baseline(
            results, dataset, dataset_name, dataset_info)

        print(f"\n\n---------- Unfairness Mitigation techniques ----------")
        results = evaluate_mitigation_techniques(
            results, dataset, dataset_name, dataset_info)

    # save the results to file
    experiment_details = {
        "DATE": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "CLASSIFIERS_HYPERPARAMETERS": CLASSIFIERS_HYPERPARAMETERS,
        "SEEDS": SEEDS,
    }

    save_results(
        filename="baseline_splits",
        results=results,
        experiment_details=experiment_details,
    )


if __name__ == "__main__":
    main()
