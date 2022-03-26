# import ipdb


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


class DIR(Preprocessing):
    def transform(self, dataset):
        return super().fit_transform(dataset)


class OptimPreproc(Preprocessing):
    pass


class Reweighing(Preprocessing):
    pass


class LFR(Preprocessing):
    pass
