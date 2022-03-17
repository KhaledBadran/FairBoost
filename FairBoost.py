## imports ##
import numpy as np
from sklearn.base import clone
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
        '''
        Preprocess data set using each pre-processing function.

                Parameters:
                        X: Features
                        y: Labels

                Returns:
                        pp_data (list): List with the different preprocessed data sets
        '''
        pp_data = []
        for ppf in self.preprocessing_functions:
            pp_data.append(ppf(X, y))
        return pp_data

    def __get_avg_dist_arr(self, data):
        '''
        For each instance in the initial data set, compute the average distance between the "cleaned" versions. 
                Parameters:
                        data (np.array): The concatenated X,y 

                Returns:
                        dist_arr (np.array): Array with the distance for each instance of each "cleaned" data set.
        '''
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
        return dist_arr

    # Adds y to the last column of X for a list of (X,y)
    def __merge_Xy(self, datasets):
        '''
        Returns instances where the last feature is the label. 
                Parameters:
                        datasets (list): List with X and y pairs.

                Returns:
                        res (np.array): List with concatenated X and y.
        '''
        res = []
        for dataset in datasets:
            X, y = dataset[0], np.expand_dims(dataset[1], axis=-1)
            m = np.concatenate([X, y], axis=-1)
            res.append(m)
        return np.array(res)

    def __unmerge_Xy(self, datasets):
        '''
        Return X,y from a dataset where y is the last column 
                Parameters:
                        datasets (list): List with concatenated X and y. 

                Returns:
                        res (list<np.array>): List with X and y pairs.
        '''
        res = []
        for dataset in datasets:
            res.append((dataset[:, :-1], dataset[:, -1]))
        return res

    def __initialize_bootstrap_datasets(self, datasets):
        '''
        Assign each instance of a data set to one of the bootstrap data sets. 
                Parameters:
                        datasets (list<np.array>): List with concatenated X and y. 

                Returns:
                        bootstrap_datasets (list<np.array>): List with the bootstrap data sets.
        '''
        bootstrap_datasets = []
        # Generate indexes array assigns each instance to
        indexes = [i for i in range(len(self.preprocessing_functions))]
        indexes = np.random.choice(
            indexes, size=len(datasets[0]), replace=True)
        for i, dataset in enumerate(datasets):
            bootstrap_datasets.append(dataset[indexes == i])
        return bootstrap_datasets

    def __fill_boostrap_datasets(self, bootstrap_datasets, datasets, p_arrays):
        '''
        Fills the bootstrap data set to the desired size. 
                Parameters:
                        bootstrap_datasets (list<np.array>): List with the bootstrap data sets.
                        datasets (list<np.array>): List with concatenated X and y. 
                        p_arrays(list<np.array>): Probability of an instance to be picked in the bootstrap process.

                Returns:
                        bootstrap_datasets (list): List with the bootstrap data sets.
        '''
        required_size = int(self.bootstrap_size*len(datasets[0]))
        for i, p in enumerate(p_arrays):
            crnt_size = len(bootstrap_datasets[i])
            dataset = datasets[i]
            indexes = [i for i in range(len(dataset))]
            indexes = np.random.choice(
                indexes, size=(required_size-crnt_size), replace=True, p=p)
            bootstrap_datasets[i] = np.concatenate(
                (bootstrap_datasets[i], dataset[indexes]))
        return bootstrap_datasets

    def __normalize(self, arrays):
        '''
        Does a array-wise normalization of values/ 
                Parameters:
                        arrays (np.array): A "list" of arrays to normalize.

                Returns:
                        n_arr (np.array): Normalized arrays.
        '''
        n_arr = []
        for arr in arrays:
            s = np.sum(arr)
            n = arr/s
            n_arr.append(n)
        return np.array(n_arr)

    # Generate the boostrap data sets
    # Returns a list of (X,y)
    def __bootstrap_datasets(self, X, y):
        '''
        Generates the bootstrap data sets for bagging. The bootstrap process depends on the self.bootstrap_type attribute.
                Parameters:
                        X: features
                        y: labels

                Returns:
                        bootstrap_datasets (list<np.array>): The bootstrap data sets
        '''
        datasets = self.__preprocess_data(X, y)
        datasets = self.__merge_Xy(datasets)
        # If we do the custom bootstrapping, we must define a custom PDF
        if self.bootstrap_type == Bootstrap_type.CUSTOM:
            dist_arrays = self.__get_avg_dist_arr(datasets)
            dist_arrays = self.__normalize(dist_arrays)
        else:
            dist_arrays = [None for _ in range(len(datasets))]

        bootstrap_datasets = self.__initialize_bootstrap_datasets(datasets)
        bootstrap_datasets = self.__fill_boostrap_datasets(
            bootstrap_datasets, datasets, dist_arrays)

        bootstrap_datasets = self.__unmerge_Xy(bootstrap_datasets)
        return bootstrap_datasets

    def fit(self, X, y):
        '''
        Fit function as per Sklearn API.
                Parameters:
                        X: features
                        y: labels

                Returns:
                        self
        '''
        datasets = self.__bootstrap_datasets(X, y)
        for X_bootstrap, y_bootstrap in datasets:
            model = clone(self.model)
            model.fit(X_bootstrap, y_bootstrap)
            self.models.append(model)
        return self

    def predict(self, X):
        '''
        Predict function as per Sklearn API.
                Parameters:
                        X: features

                Returns:
                        y_pred: Predicted labels.
        '''
        y_pred = []
        for i in range(len(self.models)):
            y_pred.append(self.models[i].predict(X))
        # Computing a soft majority voting
        y_pred = np.array(y_pred).transpose()
        y_pred = np.mean(y_pred, axis=-1).astype(int)
        return y_pred
