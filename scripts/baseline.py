# Sources:
# https://github.com/Trusted-AI/AIF360/blob/master/examples/tutorial_medical_expenditure.ipynb
# https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_meta_classifier.ipynb
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

import sys

from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.model_selection import GridSearchCV

sys.path.insert(0, '../')

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display

# Datasets
from aif360.datasets import AdultDataset
from aif360.datasets import CompasDataset
from aif360.datasets import GermanDataset
from aif360.sklearn.datasets import fetch_german

# Fairness metrics
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric

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

# Bias mitigation techniques
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover, LFR, OptimPreproc

from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from constants import DATASETS

np.random.seed(0)


def main():
    # sensitive_attribute = 'sex'
    # privileged_groups = [{'sex': 1}]
    # unprivileged_groups = [{'sex': 0}]

    # german_dataset = load_preproc_data_german(['sex'])
    # adult_dataset = load_preproc_data_adult(['sex'])
    # compas_dataset = load_preproc_data_compas(['sex'])
    #
    #
    # datasets = {'German': german_dataset,
    #             'Adult': adult_dataset,
    #             'compas': compas_dataset
    #             }

    classifiers = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(max_depth=2, n_estimators=2, max_features=1),
    }

    for dataset_name, dataset_info in DATASETS.items():

        print(f'\n\n############## in dataset: {dataset_name} ###############\n')
        dataset = dataset_info['original_dataset']

        debaiasing_algorithms = {
            "Reweighing": Reweighing(privileged_groups=dataset_info['privileged_groups'],
                                     unprivileged_groups=dataset_info['unprivileged_groups']),
            "DisparateImpactRemover": DisparateImpactRemover(sensitive_attribute=dataset_info['sensitive_attribute']),
            # 'OptimPreproc': OptimPreproc(OptTools, dataset_info['optim_options'],
            #                              unprivileged_groups=dataset_info['unprivileged_groups'],
            #                              privileged_groups=dataset_info['privileged_groups'])
            # 'LFR': LFR(privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)
        }

        # Apply standard scaler
        # TODO: fit scaler on train and apply on test
        dataset.features = StandardScaler().fit_transform(dataset.features)

        # split the data
        train_split, test_split = dataset.split([0.7], shuffle=True)

        # get X and y for training and test splits
        X_train, y_train = train_split.features, train_split.labels.ravel()
        X_test, y_test = test_split.features, test_split.labels.ravel()

        # Train model
        for clf_name, clf in classifiers.items():
            print(f'\nevaluating classifier {clf_name}')
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f'accuracy {accuracy}')

            # parity gap
            # fairness_metric = average_odds_error(y_test, y_pred, prot_attr='sex')
            # print(f"fairness {fairness_metric}")

            # Apply debaiasing algorithm
            for debaiasing_algo_name, debaiasing_algo in debaiasing_algorithms.items():
                # RW = Reweighing(privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)

                if debaiasing_algo_name == "OptimPreproc":
                    # Transform training data and align features
                    debaiasing_algo = debaiasing_algo.fit(train_split)
                    # Transform training data and align features
                    train_split_transformed = debaiasing_algo.transform(train_split, transform_Y=True)
                    train_split_transformed = train_split.align_datasets(train_split_transformed)
                else:
                    # TODO: apply transformation on test split
                    train_split_transformed = debaiasing_algo.fit_transform(train_split)

                X_train, y_train = train_split_transformed.features, train_split_transformed.labels.ravel()

                # X_train = RW.transform(X_train)
                print(f'\nevaluating classifier {clf_name} after {debaiasing_algo_name}')
                clf.fit(X_train, y_train)

                y_pred = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f'accuracy {accuracy} (with {debaiasing_algo_name})')


if __name__ == "__main__":
    main()
