from enum import Enum
import ipdb


class Bootstrap_type(Enum):
    NONE = 1
    DEFAULT = 2
    CUSTOM = 3


class Preprocessing:
    def __init__(self, preprocessing, transform_params={}):
        self.preprocessing = preprocessing
        self.transform_params = transform_params
        # ipdb.set_trace(context=6)

    def __str__(self) -> str:
        return f'{type(self.preprocessing).__name__}'

    def fit_transform(self, dataset):
        d = self.preprocessing.fit_transform(dataset, **self.transform_params)
        return d.features, d.labels, d.instance_weights

    def transform(self, dataset):
        d = self.preprocessing.transform(dataset, **self.transform_params)
        return d.features, d.labels, d.instance_weights
