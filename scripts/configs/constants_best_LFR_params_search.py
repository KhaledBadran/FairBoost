from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .utils import get_LFR_hyperparameters_search, initialize_adult_dataset, initialize_compass_dataset, initialize_german_dataset
from .enums import Dataset_name

# TODO: run LFR best param analysis with more seeds
SEEDS = [1]

# ------------------------------- DATASET HYPERPARAMETERS ------------------------------- #
DATASETS = {
    Dataset_name.GERMAN: {
        "sensitive_attribute": "sex",
        "privileged_groups": [{"sex": 1}],
        "unprivileged_groups": [{"sex": 0}],
        "original_dataset": initialize_german_dataset(),
        "hyperparams": get_LFR_hyperparameters_search(dataset_name=Dataset_name.GERMAN),
    },
    Dataset_name.ADULT: {
        "sensitive_attribute": "sex",
        "privileged_groups": [{"sex": 1}],
        "unprivileged_groups": [{"sex": 0}],
        "original_dataset": initialize_adult_dataset(),
        "hyperparams": get_LFR_hyperparameters_search(dataset_name=Dataset_name.ADULT),
    },
    Dataset_name.COMPASS: {
        "sensitive_attribute": "sex",
        "privileged_groups": [{"sex": 1}],
        "unprivileged_groups": [{"sex": 0}],
        "original_dataset": initialize_compass_dataset(),
        "hyperparams": get_LFR_hyperparameters_search(dataset_name=Dataset_name.COMPASS),
    },
}

# ------------------------------- CLASSIFIERS GRID ------------------------------- #
CLASSIFIERS_HYPERPARAMETERS = {
    "Logistic Regression": {'max_iter': 5000},
    "Random Forest": {"max_depth": 10, "n_estimators": 5, "max_features": 2},
}

CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(
        **CLASSIFIERS_HYPERPARAMETERS["Logistic Regression"]
    ),
    "Random Forest": RandomForestClassifier(
        **CLASSIFIERS_HYPERPARAMETERS["Random Forest"]
    ),
}
