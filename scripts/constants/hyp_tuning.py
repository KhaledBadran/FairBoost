# optim_option values is from https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_optim_data_preproc.ipynb

from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions import (
    get_distortion_adult,
    get_distortion_german,
    get_distortion_compas,
)
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import (
    load_preproc_data_adult,
    load_preproc_data_german,
    load_preproc_data_compas,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid

from FairBoost.main import Bootstrap_type


# ------------------------------- DATASET HYPERPARAMETERS ------------------------------- #
DATASETS = {
    "german": {
        # if changed, apply changes to to the original_dataset attribute
        "sensitive_attribute": "sex",
        "privileged_groups": [{"sex": 1}],
        "unprivileged_groups": [{"sex": 0}],
        "original_dataset": load_preproc_data_german(["sex"]),
        "optim_options": {
            "distortion_fun": get_distortion_german,
            "epsilon": 0.1,
            "clist": [0.99, 1.99, 2.99],
            "dlist": [0.1, 0.05, 0],
        },
    },
    "adult": {
        "sensitive_attribute": "sex",
        "privileged_groups": [{"sex": 1}],
        "unprivileged_groups": [{"sex": 0}],
        "original_dataset": load_preproc_data_adult(["sex"]),
        "optim_options": {
            "distortion_fun": get_distortion_adult,
            "epsilon": 0.05,
            "clist": [0.99, 1.99, 2.99],
            "dlist": [0.1, 0.05, 0],
        },
    },
    "compas": {
        "sensitive_attribute": "sex",
        "privileged_groups": [{"sex": 1}],
        "unprivileged_groups": [{"sex": 0}],
        "original_dataset": load_preproc_data_compas(["sex"]),
        "optim_options": {
            "distortion_fun": get_distortion_compas,
            "epsilon": 0.05,
            "clist": [0.99, 1.99, 2.99],
            "dlist": [0.1, 0.05, 0],
        },
    },
}

# ------------------------------- CLASSIFIERS GRID ------------------------------- #
CLASSIFIERS_HYPERPARAMETERS = {
    "Logistic Regression": {},
    "Random Forest": {'max_depth': 10, 'n_estimators': 5, 'max_features': 2},
}

CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(**CLASSIFIERS_HYPERPARAMETERS['Logistic Regression']),
    "Random Forest": RandomForestClassifier(**CLASSIFIERS_HYPERPARAMETERS['Random Forest']),
}

# ------------------------------- HYPERPARAMETER GRIDS ------------------------------- #
DisparateImpactRemover_param_grid = {'init': list(
    ParameterGrid({'repair_level': [0.01, 0.5, 1.0]}))}

LFR_param_grid = {'init': list(ParameterGrid({"k": [5], "Ax": [0.01], "Ay": [1.0], "Az": [50.0]})),
                  'transform': list(ParameterGrid({"threshold": [0.01, 0.5, 0.1]}))}

FairBoost_param_grid = {
    'bootstrap_type': [Bootstrap_type.NONE, Bootstrap_type.DEFAULT, Bootstrap_type.CUSTOM]
}


# ------------------------------- TOP-LEVEL HYPERPARAMETER GRIDS ------------------------------- #
HYPERPARAMETERS = {
    "Reweighing": [{}],
    "DisparateImpactRemover": list(ParameterGrid(DisparateImpactRemover_param_grid)),
    "OptimPreproc": [{}],
    "LFR": list(ParameterGrid(LFR_param_grid)),
}
FAIRBOOST_HYPERPARAMETERS = {
    'preprocessing': list(ParameterGrid(HYPERPARAMETERS)),
    'init': list(ParameterGrid(FairBoost_param_grid))
}
