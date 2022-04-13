## imports ##
import numpy as np
from sklearn.base import clone
import scipy.spatial.distance as dist
from typeguard import typechecked
from enum import Enum
from aif360.datasets import BinaryLabelDataset
from scipy.special import softmax
from typing import List, Tuple
# import  ipdb

from FairBoost.wrappers import Preprocessing
from .utils import quiet


class Bootstrap_type(str, Enum):
    NONE = "NONE"
    DEFAULT = 'DEFAULT'
    CUSTOM = 'CUSTOM'


@typechecked
class FairBoost(object):
    def __init__(self, model, preprocessing_functions: List[Preprocessing], bootstrap_type=Bootstrap_type.DEFAULT, bootstrap_size=1, verbose=False):
        self.model = model
        self.preprocessing_functions = preprocessing_functions
        self.n_elements = len(preprocessing_functions)
        self.bootstrap_size = bootstrap_size
        self.bootstrap_type = bootstrap_type
        self.verbose = verbose

        # The trained models
        self.models = []
        # TODO: consider other distance functions
        self.dist_func = dist.cosine
        # ipdb.set_trace(context=6)

    def __transform(self, dataset: BinaryLabelDataset, fit=False) -> List[Tuple]:
        '''
        Preprocess data set using each pre-processing function.

                Parameters:
                        dataset

                Returns:
                        pp_data (list): List with the different preprocessed data sets
        '''
        pp_data = []
        if self.verbose:
            print(f'Transforming data set with:')
        for ppf in self.preprocessing_functions:
            if self.verbose:
                print(f'\t-{ppf}')
            # Call fit_transform or transform depending on Fairboost stage
            func = ppf.fit_transform if fit else ppf.transform
            d = quiet(func, [dataset.copy(deepcopy=True)])
            pp_data.append(d)
        return pp_data

    def __get_avg_dist_arr(self, datasets: np.array) -> np.array:
        '''
        For each instance in the initial data set, compute the average distance between the "cleaned" versions. 
                Parameters:
                        X (np.array): features

                Returns:
                        dist_arr (np.array): Array with the distance for each instance of each "cleaned" data set.
        '''
        # Remove weights from dataset
        datasets = datasets[:, :, :-1]
        # Swap the first two dimensions so we iterate over instances instead of data sets
        datasets = datasets.transpose([1, 0, 2])
        # Initializing the average distances array
        dist_arr = np.zeros(
            shape=(len(datasets), len(self.preprocessing_functions)))
        # Fill the avg distances array
        for i, pp_instances in enumerate(datasets):
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

    def __merge_Xyw(self, datasets: List[Tuple]) -> np.array:
        '''
        Returns instances where the last feature is the label. 
                Parameters:
                        datasets: List with X, y and weight pairs.

                Returns:
                        res: List with concatenated X and y.
        '''
        res = []
        for dataset in datasets:
            X, y, w = dataset[0], dataset[1], np.expand_dims(
                dataset[2], axis=-1)
            m = np.concatenate([X, y, w], axis=-1)
            res.append(m)
        return np.array(res)

    def __unmerge_Xy(self, datasets: List[np.array]) -> List[Tuple]:
        '''
        Return X,y from a dataset where y is the last column 
                Parameters:
                        datasets: List with concatenated X and y. 

                Returns:
                        res: List with X, y and weight pairs.
        '''
        res = []
        for dataset in datasets:
            res.append((dataset[:, :-2], dataset[:, -2], dataset[:, -1]))
        return res

    def __prefill_bootstrap_datasets(self, datasets: np.array) -> List[np.array]:
        '''
        Assign each instance of a data set to one of the bootstrap data sets. 
                Parameters:
                        datasets: List with concatenated X, y and weights. 

                Returns:
                        bootstrap_datasets: List with the bootstrap data sets.
        '''
        bootstrap_datasets = []
        # Generate indexes array assigns each instance to
        indexes = [i for i in range(len(self.preprocessing_functions))]
        indexes = np.random.choice(
            indexes, size=len(datasets[0]), replace=True)
        for i, dataset in enumerate(datasets):
            bootstrap_datasets.append(dataset[indexes == i])
        return bootstrap_datasets

    def __fill_boostrap_datasets(self, bootstrap_datasets: List[np.array], datasets: np.array, p_arrays: List) -> List[np.array]:
        '''
        Fills the bootstrap data set to the desired size. 
                Parameters:
                        bootstrap_datasets: List with the bootstrap data sets.
                        datasets:           List with concatenated X, y and w. 
                        p_arrays:           Probability of an instance to be picked in the bootstrap process.

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
                (bootstrap_datasets[i], dataset[indexes])) if len(bootstrap_datasets[i]) > 0 else dataset[indexes]
        return bootstrap_datasets

    def __get_p_arrays(self, datasets) -> List:
        '''
        Generates the probability distributions for the bootstrapping
        process. If we use CUSTOM bootstrapping, instances that differs
        a lot between preprocess datasets will have higher chances of
        being picked up. If we use DEFAULT boostrapping, simply return 
        a None array, which is the equivalent of a uniform distribution.
                Parameters:
                        datasets: The dataset that will be used for boostrapping.
                Returns:
                        p_arrays (list<np.array>): An array of probability distribution per dataset.
        '''
        p_arrays = [None for _ in range(len(datasets))]
        if self.bootstrap_type == Bootstrap_type.CUSTOM:
            p_arrays = self.__get_avg_dist_arr(datasets)
            p_arrays = softmax(p_arrays, axis=1).tolist()
        return p_arrays

    def __get_bootstrap_datasets(self, datasets: np.array, p_array=[]) -> List:
        '''
        Generates boostrap datasets using the given datasets. 
        The boostrapping process uses the probability distribution
        specified in the p_array.
                Parameters:
                        datasets: The dataset used for boostrapping.
                        p_array: An array containing the probability each instance
                                 is picked up in the bootstrapping process.
                Returns:
                        bootstrap_datasets (list<np.array>): The bootstrap datasets
        '''
        b_datasets = [np.array([]) for _ in range(len(datasets))]
        if self.bootstrap_type == Bootstrap_type.CUSTOM:
            b_datasets = self.__prefill_bootstrap_datasets(datasets)

        b_datasets = self.__fill_boostrap_datasets(
            b_datasets, datasets, p_array)
        return b_datasets

    def __bootstrap_datasets(self, datasets: List[Tuple]) -> List[Tuple]:
        '''
        Generates the bootstrap datasets for bagging.
                Parameters:
                        datasets

                Returns:
                        bootstrap_datasets (list<np.array>): The bootstrap data sets
        '''
        datasets = self.__merge_Xyw(datasets)
        p_arrays = self.__get_p_arrays(datasets)
        b_datasets = self.__get_bootstrap_datasets(datasets, p_arrays)
        b_datasets = self.__unmerge_Xy(b_datasets)
        return b_datasets

    def fit(self, dataset: BinaryLabelDataset):
        '''
        Fit function as per Sklearn API.
                Parameters:
                        dataset

                Returns:
                        self
        '''
        datasets = self.__transform(dataset, fit=True)
        if self.bootstrap_type != Bootstrap_type.NONE:
            datasets = self.__bootstrap_datasets(datasets)
        for X_bootstrap, y_bootstrap, w in datasets:
            y_bootstrap = y_bootstrap.ravel()
            model = clone(self.model)
            model.fit(X_bootstrap, y_bootstrap, sample_weight=w)
            self.models.append(model)
        return self

    def predict(self, dataset: BinaryLabelDataset) -> np.array:
        '''
        Predict function as per Sklearn API.
                Parameters:
                        dataset

                Returns:
                        y_pred: Predicted labels.
        '''
        y_pred = []
        datasets = self.__transform(dataset)
        for i in range(len(self.models)):
            X, y, _ = datasets[i]
            y_pred.append(self.models[i].predict(X))
        # Computing a soft majority voting
        y_pred = np.array(y_pred).transpose()
        y_pred = np.mean(y_pred, axis=-1).astype(int)
        return y_pred
