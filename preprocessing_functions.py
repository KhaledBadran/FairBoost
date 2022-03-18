import numpy as np

from aif360.algorithms.preprocessing import Reweighing, OptimPreproc
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.algorithms.preprocessing.lfr import LFR

# Functions that applies the preprocessing
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools


def reweighing(data, unprivileged_groups, privileged_groups, dataset_object):
    dataset_object.features = data
    RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    dataset_transf_train = RW.fit_transform(dataset_object)
    return [dataset_transf_train.features, dataset_transf_train.labels.ravel()]


def dir(data, protected_attributes, dataset_object, repair_level=1):
    dataset_object.features = data
    index = dataset_object.feature_names.index(protected_attributes[0])
    di = DisparateImpactRemover(repair_level=repair_level)
    transformed_data = di.fit_transform(dataset_object)
    transformed_data.features = np.delete(transformed_data.features, index, axis=1)
    return [transformed_data.features, transformed_data.labels.ravel()]


def lfr(data, unprivileged_groups, privileged_groups, dataset_object, k=10, Ax=0.1, Ay=1.0, Az=2.0, verbose=0,
        maxiter=5000,
        maxfun=5000):
    # TODO check if we should scale the data
    # scale_orig = StandardScaler()
    # dataset_orig_train.features = scale_orig.fit_transform(dataset_orig_train.features)
    dataset_object.features = data
    TR = LFR(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups, k=k, Ax=Ax, Ay=Ay,
             Az=Az, verbose=verbose)
    TR = TR.fit(dataset_object, maxiter, maxfun)
    transformed_data = TR.transform(dataset_object)
    return [transformed_data.features, transformed_data.labels.ravel()]


def optimized_preprocessing(data, unprivileged_groups, privileged_groups, dataset_object, optim_options):
    # TODO check if we should scale the data
    # scale_orig = StandardScaler()
    # dataset_orig_train.features = scale_orig.fit_transform(dataset_orig_train.features)
    dataset_object.features = data
    OP = OptimPreproc(OptTools, optim_options,
                      unprivileged_groups=unprivileged_groups,
                      privileged_groups=privileged_groups)

    OP = OP.fit(dataset_object)

    # Transform training data and align features
    transformed_data = OP.transform(dataset_object, transform_Y=True)
    transformed_data = dataset_object.align_datasets(transformed_data)
    return [transformed_data.features, transformed_data.labels.ravel()]


# Functions to generate lambda function
def generate_lambda_function_reweighing(dataset_object, unprivileged_groups, privileged_groups):
    return lambda data: reweighing(data, unprivileged_groups, privileged_groups, dataset_object)


def generate_lambda_function_dir(dataset_object, protected_attributes):
    return lambda data: dir(data, protected_attributes, dataset_object)


def generate_lambda_function_LFR(dataset_object, unprivileged_groups, privileged_groups):
    return lambda data: lfr(data, unprivileged_groups, privileged_groups, dataset_object)


def generate_lambda_function_optimized_preprocessing(dataset_object, unprivileged_groups, privileged_groups, optim_options):
    return lambda data: optimized_preprocessing(data, unprivileged_groups, privileged_groups, dataset_object, optim_options)
