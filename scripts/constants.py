# optim_option values is from https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_optim_data_preproc.ipynb

from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions\
            import get_distortion_adult, get_distortion_german, get_distortion_compas
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
        import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

DATASETS = {
    'german': {
        'sensitive_attribute' : 'sex',     # if changed, apply changes to to the original_dataset attribute
        'privileged_groups' : [{'sex': 1}],
        'unprivileged_groups' : [{'sex': 0}],
        'original_dataset':  load_preproc_data_german(['sex']),
        'optim_options': {
            "distortion_fun": get_distortion_german,
            "epsilon": 0.1,
            "clist": [0.99, 1.99, 2.99],
            "dlist": [.1, 0.05, 0]
        },

    },
    'adult': {
        'sensitive_attribute': 'sex',
        'privileged_groups': [{'sex': 1}],
        'unprivileged_groups': [{'sex': 0}],
        'original_dataset': load_preproc_data_adult(['sex']),
        'optim_options': {
            "distortion_fun": get_distortion_adult,
            "epsilon": 0.05,
            "clist": [0.99, 1.99, 2.99],
            "dlist": [.1, 0.05, 0]
        }
    },
    'compas': {
        'sensitive_attribute': 'sex',
        'privileged_groups': [{'sex': 1}],
        'unprivileged_groups': [{'sex': 0}],
        'original_dataset': load_preproc_data_compas(['sex']),
        'optim_options': {
            "distortion_fun": get_distortion_compas,
            "epsilon": 0.05,
            "clist": [0.99, 1.99, 2.99],
            "dlist": [.1, 0.05, 0]
        }
    }
}


CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(max_depth=10, n_estimators=5, max_features=2)
}

HYPERPARAMETERS = {
    "Reweighing": [{}],
    "DisparateImpactRemover": [{'threshold': 0.01}, {'threshold': 0.5}, {'threshold': 0.5}],
    "OptimPreproc": [{}],
    "LFR": [{}],
}