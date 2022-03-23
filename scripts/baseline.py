# Sources:
# https://github.com/Trusted-AI/AIF360/blob/master/examples/tutorial_medical_expenditure.ipynb
# https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_meta_classifier.ipynb
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

import sys

from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display
import pandas as pd

# Datasets
from aif360.datasets import AdultDataset
from aif360.datasets import CompasDataset
from aif360.datasets import GermanDataset
from aif360.sklearn.datasets import fetch_german

# Fairness metrics
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.datasets import BinaryLabelDataset

# Explainers
from aif360.explainers import MetricTextExplainer

# Scalers
from sklearn.preprocessing import StandardScaler

# Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from aif360.sklearn.metrics import disparate_impact_ratio, average_odds_error
import pandas as pd
import json

# Bias mitigation techniques
from aif360.algorithms.preprocessing import (
    Reweighing,
    DisparateImpactRemover,
    LFR,
    OptimPreproc,
)

from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from constants import DATASETS, CLASSIFIERS

from collections import defaultdict

np.random.seed(0)


def train_test_models(train_dataset, test_dataset, dataset_info):
    results = defaultdict(dict)

    # Train models
    X_train, y_train = train_dataset.features, train_dataset.labels.ravel()
    X_test, y_test = test_dataset.features, test_dataset.labels.ravel()

    for clf_name, clf in CLASSIFIERS.items():
        print(f"\nevaluating classifier {clf_name}")
        clf.fit(X_train, y_train, sample_weight=train_dataset.instance_weights)

        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"accuracy {accuracy}")

        classified_dataset = test_dataset.copy()
        classified_dataset.labels = y_pred
        classification_metric = ClassificationMetric(
            dataset=test_dataset,
            classified_dataset=classified_dataset,
            unprivileged_groups=dataset_info["unprivileged_groups"],
            privileged_groups=dataset_info["privileged_groups"],
        )
        print(f"disparate impact {classification_metric.disparate_impact()}")
        print(
            f"average odds difference {classification_metric.average_odds_difference()}"
        )

        results[clf_name] = {
            "accuracy": accuracy,
            "disparate_impact": classification_metric.disparate_impact(),
            "average_odds_difference": classification_metric.average_odds_difference(),
        }

    return dict(results)

# @typehceck
def initial_preprocessing(train_dataset: BinaryLabelDataset, test_dataset):
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


def apply_preprocessing_algo(
    algo_name,
    transformer,
    train_dataset: BinaryLabelDataset,
    test_dataset: BinaryLabelDataset,
    dataset_info,
):

    if algo_name == "Reweighing":
        # RW = Reweighing(privileged_groups=dataset_info['privileged_groups'],
        #                 unprivileged_groups=dataset_info['unprivileged_groups'])
        train_dataset_transformed = transformer.fit_transform(train_dataset)
        test_dataset_transformed = test_dataset.copy()

    elif algo_name == "DisparateImpactRemover":
        # DIR = DisparateImpactRemover(sensitive_attribute=dataset_info['sensitive_attribute'])

        index = train_dataset.feature_names.index(dataset_info["sensitive_attribute"])

        train_dataset_transformed = transformer.fit_transform(train_dataset)
        test_dataset_transformed = transformer.fit_transform(test_dataset)

        # delete protected columns
        train_dataset_transformed.features = np.delete(
            train_dataset_transformed.features, index, axis=1
        )
        test_dataset_transformed.features = np.delete(
            test_dataset_transformed.features, index, axis=1
        )

    return (
        train_dataset_transformed,
        test_dataset_transformed,
    )


def initialize_debaiasing_algorithms(dataset_info):
    return {
        "Reweighing": Reweighing(
            privileged_groups=dataset_info["privileged_groups"],
            unprivileged_groups=dataset_info["unprivileged_groups"],
        ),
        "DisparateImpactRemover": DisparateImpactRemover(
            sensitive_attribute=dataset_info["sensitive_attribute"]
        ),
        # "OptimPreproc": OptimPreproc(
        #     OptTools,
        #     dataset_info["optim_options"],
        #     unprivileged_groups=dataset_info["unprivileged_groups"],
        #     privileged_groups=dataset_info["privileged_groups"],
        # )
        # 'LFR': LFR(privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)
    }


def main():

    results = defaultdict(dict)

    for dataset_name, dataset_info in DATASETS.items():

        print(f"\n\n$$$$in dataset {dataset_name}$$$$$\n")

        dataset = dataset_info["original_dataset"]

        debaiasing_algorithms = initialize_debaiasing_algorithms(
            dataset_info=dataset_info
        )

        train_split, test_split = dataset.split([0.7], shuffle=True)
        train_split, test_split = initial_preprocessing(train_split, test_split)
        results[dataset_name]["baseline"] = train_test_models(
            train_split, test_split, dataset_info=dataset_info
        )

        for (debaiasing_algo_name,debaiasing_transformer,) in debaiasing_algorithms.items():
            print(f"\n\n####After applying {debaiasing_algo_name}######\n")

            train_split_transformed, test_split_transformed = apply_preprocessing_algo(
                debaiasing_algo_name,
                debaiasing_transformer,
                train_split,
                test_split,
                dataset_info,
            )

            results[dataset_name][debaiasing_algo_name] = train_test_models(
                train_split_transformed,
                test_split_transformed,
                dataset_info=dataset_info,
            )

    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()