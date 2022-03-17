from sklearn.tree import DecisionTreeClassifier
import numpy as np

from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.preprocessing import DisparateImpactRemover

from fairboost import FairBoost
from data import get_german_dataset
from preprocessing_functions import generate_lambda_function_dir, generate_lambda_function_reweighing, \
    generate_lambda_function_LFR

dataset_orig_train, dataset_orig_val, dataset_orig_test, unprivileged_groups, privileged_groups = get_german_dataset()
X = dataset_orig_train.features
y = dataset_orig_train.labels.ravel()

model = DecisionTreeClassifier(class_weight='balanced')

preprocessing1 = generate_lambda_function_reweighing(unprivileged_groups, privileged_groups)
preprocessing2 = generate_lambda_function_dir(dataset_orig_train.protected_attribute_names)
preprocessing3 = generate_lambda_function_LFR(unprivileged_groups, privileged_groups)
preprocessing = (preprocessing1, preprocessing2, preprocessing3)

# data = {'X': X, 'y': y}
ens = FairBoost(model, preprocessing)

## train the ensemble & view estimates for prediction error ##
ens.fit(dataset_orig_train, unprivileged_groups, privileged_groups)
ens.predict(dataset_orig_test)
