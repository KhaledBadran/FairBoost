import random
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from aif360.datasets import GermanDataset
from aif360.algorithms.preprocessing import Reweighing, OptimPreproc, LFR, DisparateImpactRemover
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions import get_distortion_german, get_distortion_adult
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_german, load_preproc_data_adult
from copy import deepcopy
import numpy as np
# from test import test_LFR

from FairBoost.main import FairBoost, Bootstrap_type
from FairBoost import wrappers


np.random.seed(42)
dataset_orig = load_preproc_data_adult()
dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)


# ------------- OptimPreproc -------------
optim_options = {
    "distortion_fun": get_distortion_adult,
    "epsilon": 0.05,
    "clist": [0.99, 1.99, 2.99],
    "dlist": [.1, 0.05, 0]
}
privileged_groups = [{'sex': 1.0}]
unprivileged_groups = [{'sex': 0.0}]
pp2 = OptimPreproc(OptTools, optim_options,
                   unprivileged_groups=unprivileged_groups,
                   privileged_groups=privileged_groups)
pp2 = wrappers.OptimPreproc(pp2)

# ------------- Reweighing -------------
pp1 = Reweighing(unprivileged_groups=unprivileged_groups,
                 privileged_groups=privileged_groups)
pp1 = wrappers.Reweighing(pp1)

# ------------- DisparateImpactRemover -------------
pp4 = DisparateImpactRemover(repair_level=.5)
pp4 = wrappers.DIR(pp4)


# ------------- Training FairBoost -------------
pp = [pp1, pp2, pp4]
model = LogisticRegression()
ens = FairBoost(model, pp, bootstrap_type=Bootstrap_type.CUSTOM)
ens = ens.fit(dataset_orig_train)
y_pred = ens.predict(dataset_orig_test)
accuracy_score(y_pred, dataset_orig_test.labels)
