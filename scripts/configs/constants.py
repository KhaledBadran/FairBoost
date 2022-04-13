from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from FairBoost.main import Bootstrap_type
from .utils import get_preproc_hyperparameters, initialize_adult_dataset, initialize_compass_dataset, initialize_german_dataset
from .enums import Dataset_name, Preproc_name

SEEDS = [1]

# ------------------------------- FAIRBOOST GRID ------------------------------- #
FairBoost_param_grid = {
    'init': {"bootstrap_type": [
        # Bootstrap_type.NONE,
        # Bootstrap_type.DEFAULT,
        Bootstrap_type.CUSTOM,
    ]},
    'preprocessing': [  # [Preproc_name.OptimPreproc],
        #   [Preproc_name.LFR],
        #   [Preproc_name.Reweighing],
        #   [Preproc_name.OptimPreproc, Preproc_name.LFR],
        # [Preproc_name.OptimPreproc, Preproc_name.Reweighing],
        #   [Preproc_name.LFR, Preproc_name.Reweighing],
        [Preproc_name.Reweighing, Preproc_name.LFR, Preproc_name.OptimPreproc]
    ]
}


# ------------------------------- DATASET HYPERPARAMETERS ------------------------------- #
DATASETS = {
    Dataset_name.GERMAN: {
        "sensitive_attribute": "sex",
        "privileged_groups": [{"sex": 1}],
        "unprivileged_groups": [{"sex": 0}],
        "original_dataset": initialize_german_dataset(),
        "hyperparams": get_preproc_hyperparameters(dataset_name=Dataset_name.GERMAN),
    },
    Dataset_name.ADULT: {
        "sensitive_attribute": "sex",
        "privileged_groups": [{"sex": 1}],
        "unprivileged_groups": [{"sex": 0}],
        "original_dataset": initialize_adult_dataset(),
        "hyperparams": get_preproc_hyperparameters(dataset_name=Dataset_name.ADULT),
    },
    Dataset_name.COMPASS: {
        "sensitive_attribute": "sex",
        "privileged_groups": [{"sex": 1}],
        "unprivileged_groups": [{"sex": 0}],
        "original_dataset": initialize_compass_dataset(),
        "hyperparams": get_preproc_hyperparameters(dataset_name=Dataset_name.COMPASS),
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
