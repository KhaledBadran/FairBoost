import numpy as np

from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.algorithms.preprocessing.lfr import LFR



# Functions that applies the preprocessing
def reweighing(data, unprivileged_groups, privileged_groups):
    RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    dataset_transf_train = RW.fit_transform(data)
    return dataset_transf_train


def dir(data, protected_attributes, repair_level=1):
    index = data.feature_names.index(protected_attributes[0])
    di = DisparateImpactRemover(repair_level=repair_level)
    transformed_data = di.fit_transform(data)
    transformed_data.features = np.delete(transformed_data.features, index, axis=1)
    return transformed_data


def lfr(data, unprivileged_groups, privileged_groups, k=10, Ax=0.1, Ay=1.0, Az=2.0, verbose=0, maxiter=5000,
        maxfun=5000):
    # check if we should scale the data
    # scale_orig = StandardScaler()
    # dataset_orig_train.features = scale_orig.fit_transform(dataset_orig_train.features)
    TR = LFR(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups, k=k, Ax=Ax, Ay=Ay,
             Az=Az, verbose=verbose)
    TR = TR.fit(data, maxiter, maxfun)
    transformed_data = TR.transform(data)
    return transformed_data


# Functions to generate lambda function
def generate_lambda_function_reweighing(unprivileged_groups, privileged_groups):
    return lambda data: reweighing(data, unprivileged_groups, privileged_groups)


def generate_lambda_function_dir(protected_attributes):
    return lambda data: dir(data, protected_attributes)


def generate_lambda_function_LFR(unprivileged_groups, privileged_groups):
    return lambda data: lfr(data, unprivileged_groups, privileged_groups)
