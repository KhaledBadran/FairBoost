# import ipdb
import numpy as np


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


class NoPreprocessing(Preprocessing):
    def __init__(self):
        super().__init__(preprocessing=None)

    def __str__(self) -> str:
        return f'NoPreprocessing'

    def fit_transform(self, dataset):
        return dataset.features, dataset.labels, dataset.instance_weights

    def transform(self, dataset):
        return dataset.features, dataset.labels, dataset.instance_weights


class DIR(Preprocessing):
    # TODO: what do we do with deletion of private attribs
    # def __delete_protected(self, dataset):
    #     index = []
    #     for protected_attribute_name in dataset.protected_attribute_names:
    #         index.append(dataset.feature_names.index(protected_attribute_name))
    #     dataset.features = np.delete(dataset.features, index, axis=1)
    #     return dataset

    def transform(self, dataset):
        return super().fit_transform(dataset)


class OptimPreproc(Preprocessing):
    def fit_transform(self, dataset):
        d = self.preprocessing.fit_transform(dataset, **self.transform_params)
        # OptimPreproc needs to align data sets after transform
        d = dataset.align_datasets(d)
        return d.features, d.labels, d.instance_weights

    def transform(self, dataset):
        d = self.preprocessing.transform(dataset, **self.transform_params)
        # OptimPreproc needs to align data sets after transform
        d = dataset.align_datasets(d)
        return d.features, d.labels, d.instance_weights


class Reweighing(Preprocessing):
    def transform(self, dataset):
        # Reweight should not reweight test data set.
        return dataset.features, dataset.labels, dataset.instance_weights


class LFR(Preprocessing):
    pass
