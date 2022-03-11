## imports ##
import numpy as np
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score


## make an ensemble classifier based on decision trees ##
class FairBoost(object):
    def __init__(self, data, model, preprocessing_functions):
        self.data = data
        self.model = model
        self.preprocessing_functions = preprocessing_functions
        self.n_elements = len(preprocessing_functions)

        # The preprocessed data
        self.preprocessed_data= []
        # The trained models
        self.models = []


    def __preprocess_data(self):
        preprocessed_data = []
        i=0
        for ppf in self.preprocessing_functions:
            X_train, X_test, y_train, y_test = train_test_split(ppf(self.data["X"]), self.data["y"], train_size=0.8)
            preprocessed_data.append({'train': (X_train, y_train), 'test': (X_test, y_test)})
            i+=1
        self.preprocessed_data = preprocessed_data
        # add filter

    def filter(self):
        pass

    def train_models(self):
        self.__preprocess_data()
        for data in self.preprocessed_data:
            model = clone(self.model)
            model.fit(data['train'][0], data['train'][1])
            self.models.append(model)

    def evaluate_models(self):
        accs = np.array([])
        pres = np.array([])
        recs = np.array([])
        for i in range(len(self.models)):
            yp = self.models[i].predict(self.preprocessed_data[i]['test'][0])
            acc = accuracy_score(self.preprocessed_data[i]['test'][1], yp)
            pre = precision_score(self.preprocessed_data[i]['test'][1], yp)
            rec = recall_score(self.preprocessed_data[i]['test'][1], yp)
            # store the error metrics
            accs = np.concatenate((accs, acc.flatten()))
            pres = np.concatenate((pres, pre.flatten()))
            recs = np.concatenate((recs, rec.flatten()))
        print(accs, pres, recs)


    def predict(self):
        predictions = []
        for i in range(len(self.models)):
            yp = self.models[i].predict(self.preprocessed_data[i]['test'][0])
            predictions.append(yp.reshape(-1, 1))
        ypred = np.round(np.mean(np.concatenate(predictions, axis=1), axis=1)).astype(int)
        acc = accuracy_score(self.preprocessed_data[0]['test'][1], ypred)
        pre = precision_score(self.preprocessed_data[0]['test'][1], ypred)
        rec = recall_score(self.preprocessed_data[0]['test'][1], ypred)
        print(acc, pre, rec)
        return (ypred)

