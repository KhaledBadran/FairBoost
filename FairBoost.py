## imports ##
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, precision_score, recall_score
import scipy.spatial.distance as dist


from enum import Enum


class Bootstrap_type(Enum):
    NONE = 1
    DEFAULT = 2
    CUSTOM = 3

# Note: FairBoost tries to replicate sklearn API


class FairBoost(object):
    def __init__(self, model, preprocessing_functions, bootstrap_type=Bootstrap_type.DEFAULT, bootstrap_size=0.63):
        self.model = model
        self.preprocessing_functions = preprocessing_functions
        self.n_elements = len(preprocessing_functions)
        self.bootstrap_size = bootstrap_size
        self.bootstrap_type = bootstrap_type

        # The trained models
        self.models = []
        # TODO: consider other distance functions
        self.dist_func = dist.cosine
        # ipdb.set_trace(context=6)

    # Generates all "cleaned" data sets
    # Returns an array of (X,y)

    def __preprocess_data(self, X, y):
        pp_data = []
        for ppf in self.preprocessing_functions:
            pp_data.append(ppf(X, y))
        return pp_data

    def __get_avg_dist_arr(self, data):
        # Swap the first two dimensions so we iterate over instances instead of data sets
        data = data.transpose([1, 0, 2])
        # Initializing the average distances array
        dist_arr = np.zeros(
            shape=(len(data), len(self.preprocessing_functions)))
        # Fill the avg distances array
        for i, pp_instances in enumerate(data):
            for j, pp_instance_j in enumerate(pp_instances):
                distances = []
                for k, pp_instance_k in enumerate(pp_instances):
                    d = self.dist_func(pp_instance_j, pp_instance_k)
                    d = np.abs(d)
                    distances.append(d)
                # One entry is zero (the distance with itself). Do not consider it in the mean.
                dist_arr[i, j] = np.sum(distances)/(len(pp_instances)-1)

        dist_arr = dist_arr.transpose([1, 0])
        # Normalize
        n_dist_arr = []
        for arr in dist_arr:
            s = np.sum(arr)
            n = arr/s
            n_dist_arr.append(n)
        return n_dist_arr

    # Adds y to the last column of X for a list of (X,y)
    def __merge_Xy(self, datasets):
        res = []
        for dataset in datasets:
            X, y = dataset[0], np.expand_dims(dataset[1], axis=-1)
            m = np.concatenate([X, y], axis=-1)
            res.append(m)
        return np.array(res)

    # Generate the boostrap data sets
    # Returns a list of (X,y)
    def __bootstrap_datasets(self, X, y):
        datasets = self.__preprocess_data(X, y)
        datasets = self.__merge_Xy(datasets)
        # If we do the custom bootstrapping, we must define a custom PDF
        if self.bootstrap_type == Bootstrap_type.CUSTOM:
            dist_arrays = self.__get_avg_dist_arr(datasets)
        else:
            dist_arrays = [None for _ in range(len(datasets))]

        bootstrap_datasets = []
        for dataset, dist_arr in zip(datasets, dist_arrays):
            indexes = [i for i in range(len(dataset))]
            size = int(self.bootstrap_size*len(dataset))
            indexes = np.random.choice(
                indexes, size=size, replace=True, p=dist_arr)
            bootstrap_datasets.append(
                (dataset[indexes, :-1], dataset[indexes, -1]))

        return bootstrap_datasets

    def fit(self, X, y):
        datasets = self.__bootstrap_datasets(X, y)
        for X_bootstrap, y_bootstrap in datasets:
            model = clone(self.model)
            model.fit(X_bootstrap, y_bootstrap)
            self.models.append(model)
        return self

    def predict(self, X):
        y_pred = []
        for i in range(len(self.models)):
            y_pred.append(self.models[i].predict(X))
        # Computing a soft majority voting
        y_pred = np.array(y_pred).transpose()
        y_pred = np.mean(y_pred, axis=-1).astype(int)
        return y_pred
