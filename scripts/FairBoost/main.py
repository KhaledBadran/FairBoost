## imports ##
import numpy as np
from sklearn.base import clone
from typeguard import typechecked
from enum import Enum
from aif360.datasets import BinaryLabelDataset
from scipy.special import softmax
from typing import List, Tuple

from FairBoost.wrappers import Preprocessing
from .utils import quiet, concat_datasets, merge_tuples, unmerge_tuples, get_avg_dist_arr


class Bootstrap_type(str, Enum):
    NONE = "NONE"
    DEFAULT = 'DEFAULT'
    CUSTOM = 'CUSTOM'


@typechecked
class FairBoost(object):
    def __init__(self, model, preprocessing_functions: List[Preprocessing], bootstrap_type=Bootstrap_type.DEFAULT, bootstrap_size=1, n_datasets=10, verbose=False):
        """
                Parameters:
                        model:  The model that will be used by Fairboost.
                                Should follow sklearn API (fit and transform functions)
                        preprocessing_functions: The unfairness mitigation techniques.
                        bootstrap_type: The type of boostraping (including not doing any).
                        bootstrap_size: The size of the bootstrap dataset proportional to the size of the datasets.
                        n_datasets: The number of bootstrap dataset generated from one dataset
                        verbose: To set Fairboost in the verbose mode.

        """
        self.model = model
        self.preprocessing_functions = preprocessing_functions
        self.n_elements = len(preprocessing_functions)
        self.bootstrap_size = bootstrap_size
        self.n_datasets = n_datasets
        self.bootstrap_type = bootstrap_type
        self.verbose = verbose
        self.training_datasets = None

        # The trained models
        self.models = []

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

    def __prefill_bootstrap_datasets(self, datasets: np.array) -> List[np.array]:
        '''
        Assign each instance of a data set to one of the bootstrap data sets.
                Parameters:
                        datasets: List with concatenated X, y and weights.

                Returns:
                        bootstrap_datasets: List with the bootstrap data sets.
        '''
        bootstrap_datasets = []
        n_datasets = len(datasets)*self.n_datasets

        # Randomly select the instances for each boostrap dataset
        indexes = [i for i in range(n_datasets)]
        indexes = np.random.choice(
            indexes, size=len(datasets[0]), replace=True)

        # Build the bootstrap datasets
        for i in range(n_datasets):
            dataset = datasets[i % len(datasets)]
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
        for i in range(len(bootstrap_datasets)):
            # Fetch the preprocessed dataset and the associated probability distribution
            p = p_arrays[i % len(p_arrays)]
            dataset = datasets[i % len(datasets)]

            # Fill the bootstrap dataset up to the desired size
            crnt_size = len(bootstrap_datasets[i])
            indexes = np.random.choice([i for i in range(len(dataset))],
                                       size=(required_size-crnt_size), replace=True, p=p)
            bootstrap_datasets[i] = concat_datasets(
                bootstrap_datasets[i], dataset[indexes])
        return bootstrap_datasets

    def __get_p_arrays(self, datasets: np.array) -> List:
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
        if self.bootstrap_type == Bootstrap_type.CUSTOM and datasets.shape[0] > 1:
            p_arrays = get_avg_dist_arr(datasets)
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
        b_datasets = [np.array([])
                      for _ in range(len(datasets)*self.n_datasets)]
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
        datasets = merge_tuples(datasets)
        p_arrays = self.__get_p_arrays(datasets)
        b_datasets = self.__get_bootstrap_datasets(datasets, p_arrays)
        b_datasets = unmerge_tuples(b_datasets)
        return b_datasets

    def get_training_datasets(self) -> List[Tuple]:
        """
        Getter of training datasets. Throws an 
        error if the training datasets have not been 
        received yet.
                Returns:
                    training_datasets
        """
        if self.training_datasets is None:
            raise Exception(
                "Fairboost has not been trained yet, thus it has no training dataset.")
        return self.training_datasets

    def fit(self, dataset: BinaryLabelDataset, preprocessed_datasets=None):
        '''
        Fit function similar to Sklearn API. If preprocessed_datasets
        parameter is given, it will skip the process of generating
        the datasets on which the models will be trained and will
        directly use preprocessed_datasets.
                Parameters:
                        dataset: The training data.
                        preprocessed_datasets:  Datasets that have already been preprocessed 
                                                and bootstrapped. These dataset are usually obtained
                                                in a previous instantiation of Fairboost.

                Returns:
                        self
        '''
        if preprocessed_datasets is None:
            datasets = self.__transform(dataset, fit=True)
            if self.bootstrap_type != Bootstrap_type.NONE:
                datasets = self.__bootstrap_datasets(datasets)
        else:
            datasets = preprocessed_datasets
        self.training_datasets = datasets
        for X_bootstrap, y_bootstrap, w in datasets:
            y_bootstrap = y_bootstrap.ravel()
            model = clone(self.model)
            model.fit(X_bootstrap, y_bootstrap, sample_weight=w)
            self.models.append(model)
        return self

    def predict(self, dataset: BinaryLabelDataset) -> np.array:
        '''
        Predict function similar to Sklearn API.
                Parameters:
                        dataset: The data used for predictions

                Returns:
                        y_pred: Predicted labels.
        '''
        y_pred = []
        datasets = self.__transform(dataset)
        for i in range(len(self.models)):
            X, y, _ = datasets[i % len(datasets)]
            y_pred.append(self.models[i].predict(X))
        # Computing a soft majority voting
        y_pred = np.array(y_pred).transpose()
        y_pred = np.mean(y_pred, axis=-1).astype(int)
        return y_pred
