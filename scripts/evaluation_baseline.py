# Sources:
# https://github.com/Trusted-AI/AIF360/blob/master/examples/tutorial_medical_expenditure.ipynb
# https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_meta_classifier.ipynb
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
import numpy as np
import pandas as pd
import json
from collections import defaultdict

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
from constants import DATASETS, CLASSIFIERS, HYPERPARAMETERS

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
    X_test, y_test = test_dataset.features, test_dataset.labels.ravel()

    for clf_name, clf in CLASSIFIERS.items():
        print(f"\nevaluating classifier {clf_name}")

        # Check if the training labels are not all the same
        if len(set(y_train)) != 1:
            clf.fit(X_train, y_train, sample_weight=train_dataset.instance_weights)

            y_pred = clf.predict(X_test)

            classified_dataset = test_dataset.copy()
            classified_dataset.labels = y_pred
            classification_metric = ClassificationMetric(
                dataset=test_dataset,
                classified_dataset=classified_dataset,
                unprivileged_groups=dataset_info["unprivileged_groups"],
                privileged_groups=dataset_info["privileged_groups"],
            )

            # calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            disparate_impact = classification_metric.disparate_impact()
            average_odds_difference = classification_metric.average_odds_difference()

            print(f"accuracy {accuracy}")
            print(f"disparate_impact {disparate_impact}")
            print(f"average odds difference {average_odds_difference}")

            results[clf_name] = {
                "accuracy": accuracy,
                "disparate_impact": disparate_impact,
                "average_odds_difference": average_odds_difference,
            }

        else:
            print(f"all training labels are same, classifier will not be trained")
            results[clf_name] = {}

    return dict(results)


@typechecked
def initial_preprocessing(
    train_dataset: BinaryLabelDataset, test_dataset: BinaryLabelDataset
) -> Tuple[BinaryLabelDataset, BinaryLabelDataset]:
    """
    Applies scaling to the dataset
    :param train_dataset: an AIF360 dataset containing the training examples with their labels
    :param test_dataset: an AIF360 dataset containing the test examples with their labels
    :return: a scaled training and test AIF360 datasets
    """
    # get X and y for training and test splits
    X_train, y_train = train_dataset.features, train_dataset.labels.ravel()
    X_test, y_test = test_dataset.features, test_dataset.labels.ravel()

    # Apply standard scaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    train_dataset_preprocessed = train_dataset.copy()
    train_dataset_preprocessed.features = X_train

    test_dataset_preprocessed = test_dataset.copy()
    test_dataset_preprocessed.features = X_test

    return train_dataset_preprocessed, test_dataset_preprocessed


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
        repair_level=hyperparameters["repair_level"],
    )
    index = train_dataset.feature_names.index(dataset_info["sensitive_attribute"])

    train_dataset_DIR = DIR.fit_transform(train_dataset)
    test_dataset_DIR = DIR.fit_transform(test_dataset)

    # delete protected columns
    train_dataset_DIR.features = np.delete(train_dataset_DIR.features, index, axis=1)
    test_dataset_DIR.features = np.delete(test_dataset_DIR.features, index, axis=1)

    return train_dataset_DIR, test_dataset_DIR


@typechecked()
def apply_OptimPreproc(
    train_dataset: BinaryLabelDataset,
    test_dataset: BinaryLabelDataset,
    dataset_info: Dict,
) -> Tuple[BinaryLabelDataset, BinaryLabelDataset]:
    """
    :param train_dataset: an AIF360 dataset containing the training examples with their labels
    :param test_dataset: an AIF360 dataset containing the test examples with their labels
    :param dataset_info: information about the dataset including privileged and unprivileged groups
    :return: a train and test datasets that have been transformed via the Optimized Preprocessing technique
    """

    OP = OptimPreproc(OptTools, dataset_info["optim_options"], verbose=False)

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
    LFR_transformer = LFR(
        unprivileged_groups=dataset_info["unprivileged_groups"],
        privileged_groups=dataset_info["privileged_groups"],
        k=hyperparameters["k"],
        Ax=hyperparameters["Ax"],
        Ay=hyperparameters["Ay"],
        Az=hyperparameters["Az"],
        verbose=0,  # Default parameters
    )

    LFR_transformer = LFR_transformer.fit(train_dataset, maxiter=5000, maxfun=5000)

    # Transform training data and align features
    train_dataset_LFR = LFR_transformer.transform(
        train_dataset, threshold=hyperparameters["threshold"]
    )
    test_dataset_LFR = LFR_transformer.transform(
        test_dataset, threshold=hyperparameters["threshold"]
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
                train_dataset, test_dataset, dataset_info
            )

        elif algo_name == "LFR":
            train_dataset_transformed, test_dataset_transformed = apply_LFR(
                train_dataset, test_dataset, dataset_info, hyperparameters
            )

    except Exception as e:
        print(f"Failed to pre-process the dataset. The error msg is:")
        print(e)

    return (
        train_dataset_transformed,
        test_dataset_transformed,
    )


def main():
    results = defaultdict(dict)

    for dataset_name, dataset_info in DATASETS.items():

        print(f"\n\n$$$$in dataset {dataset_name}$$$$$\n")

        dataset: BinaryLabelDataset = dataset_info["original_dataset"]

        train_split, test_split = dataset.split([0.7], shuffle=True, seed=0)
        # train_split, test_split = initial_preprocessing(train_split, test_split)

        results[dataset_name]["baseline"] = train_test_models(
            train_split, test_split, dataset_info=dataset_info
        )

        for (
            debaiasing_algo_name,
            hyperparameters_space,
        ) in HYPERPARAMETERS.items():
            print(f"\n\n####After applying {debaiasing_algo_name}######\n")

            results[dataset_name][debaiasing_algo_name] = []

            for hyperparameters in hyperparameters_space:
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

                performance_metrics = train_test_models(
                    train_split_transformed,
                    test_split_transformed,
                    dataset_info=dataset_info,
                )

                # record the used hyperparameters
                results[dataset_name][debaiasing_algo_name].append(
                    {"hyperparameters": hyperparameters, "results": performance_metrics}
                )
    # save the results to file

    with open("results.json", "w") as fp:
        json.dump(results, fp, indent=4)

if __name__ == "__main__":
    main()
