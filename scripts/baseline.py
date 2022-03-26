# Sources:
# https://github.com/Trusted-AI/AIF360/blob/master/examples/tutorial_medical_expenditure.ipynb
# https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_meta_classifier.ipynb
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
import copy
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
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools

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
            print(f'all training labels are same, classifier will not be trained')
            results[clf_name] = {}

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
    algo_transformer,
    train_dataset: BinaryLabelDataset,
    test_dataset: BinaryLabelDataset,
    dataset_info,
):

    # Copy the transformer to avoid bugs
    transformer = copy.deepcopy(algo_transformer)

    # Initialize the transformed datasets
    train_dataset_transformed = train_dataset.copy(deepcopy=True)
    test_dataset_transformed = test_dataset.copy(deepcopy=True)

    try:
        if algo_name == "Reweighing":
            # RW = Reweighing(privileged_groups=dataset_info['privileged_groups'],
            #                 unprivileged_groups=dataset_info['unprivileged_groups'])
            train_dataset_transformed = transformer.fit_transform(train_dataset)
            test_dataset_transformed = test_dataset.copy(deepcopy=True)

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

        elif algo_name == "OptimPreproc":

            transformer = transformer.fit(train_dataset)
            train_dataset_transformed = transformer.transform(train_dataset, transform_Y=True)

            train_dataset_transformed = train_dataset.align_datasets(train_dataset_transformed)

            test_dataset_transformed = transformer.transform(test_dataset, transform_Y=True)
            test_dataset_transformed = test_dataset.align_datasets(test_dataset_transformed)

        elif algo_name == "LFR":

            transformer = transformer.fit(train_dataset, maxiter=5000, maxfun=5000)

            # Transform training data and align features
            train_dataset_transformed = transformer.transform(train_dataset, threshold=0.35)
            test_dataset_transformed = transformer.transform(test_dataset, threshold=0.35)



    except Exception as e:
        print(f'Failed to pre-process the dataset. The error msg is:')
        print(e)

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
        "OptimPreproc": OptimPreproc(
            OptTools,
            dataset_info["optim_options"],
            verbose=False,
            # This algorithm does not use the privileged and unprivileged groups specified during initialization
            # unprivileged_groups=dataset_info["unprivileged_groups"],
            # privileged_groups=dataset_info["privileged_groups"],
        ),
        'LFR': LFR(unprivileged_groups=dataset_info['unprivileged_groups'],
                 privileged_groups=dataset_info['privileged_groups'],
                 k=5, Ax=0.01, Ay=1.0, Az=50.0, verbose=0    # Default parameters
                )
    }


def main():

    results = defaultdict(dict)

    for dataset_name, dataset_info in DATASETS.items():

        print(f"\n\n$$$$in dataset {dataset_name}$$$$$\n")

        dataset = dataset_info["original_dataset"]

        debaiasing_algorithms = initialize_debaiasing_algorithms(
            dataset_info=dataset_info
        )

        train_split, test_split = dataset.split([0.7], shuffle=True, seed=0)
        # train_split, test_split = initial_preprocessing(train_split, test_split)
        results[dataset_name]["baseline"] = train_test_models(
            train_split, test_split, dataset_info=dataset_info
        )

        for (debaiasing_algo_name, debaiasing_transformer,) in debaiasing_algorithms.items():
            print(f"\n\n####After applying {debaiasing_algo_name}######\n")

            train_split_transformed, test_split_transformed = apply_preprocessing_algo(
                debaiasing_algo_name,
                debaiasing_transformer,
                train_split.copy(deepcopy=True),
                test_split.copy(deepcopy=True),
                dataset_info,
            )

            results[dataset_name][debaiasing_algo_name] = train_test_models(
                train_split_transformed,
                test_split_transformed,
                dataset_info=dataset_info,
            )

    # print(json.dumps(results, indent=4))
    with open('results.json', 'w') as fp:
        json.dump(results, fp, indent=4)

if __name__ == "__main__":
    main()