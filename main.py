import numpy as np
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold,cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score

from FairBoost import FairBoost


def function1(data):
    return data

def function2(data):
    return data * 0.2

# def preprocessing1(data):
#   return lambda a : function1(data)

preprocessing1= lambda data:function1(data)
preprocessing2= lambda data:function2(data)

data = load_breast_cancer()
X    = data.data
y    = data.target
model = DecisionTreeClassifier(class_weight='balanced')
preprocessing = (preprocessing1, preprocessing2)
## declare an ensemble instance with default parameters ##

data = {'X': X, 'y': y}
ens = FairBoost(data, model, preprocessing)

## train the ensemble & view estimates for prediction error ##
ens.train_models()
ens.predict()


