# Fairboost
This README describes the API of Fairboost and discuss some of its details.


## API description

### init
```python
def __init__(self, model, preprocessing_functions: List[Preprocessing], bootstrap_type=Bootstrap_type.DEFAULT, bootstrap_size=1, n_datasets=10, verbose=False)
```
- model: The model that will be used by Fairboost. Should follow sklearn API (fit and transform functions). 
- preprocessing_functions: The unfairness mitigation techniques. Should follow [AFI360 preprocessing function's API](https://aif360.readthedocs.io/en/latest/modules/algorithms.html#module-aif360.algorithms.preprocessing). For more details on how to make it extensible, refer to the [Wrappers](#wrappers) section.
- bootstrap_type: The type of bootstraping. The three choices `NONE`, `DEFAULT` and `CUSTOM`. Refer to the paper for further information. Defaults to `DEFAULT`.
- bootstrap_size: The size of the bootstrap datasets proportionally to the initial dataset. Defaults to `1` (same size), [sklearn's default value](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html).
- n_datasets: The number of bootstrap datasets generated from one dataset. Defaults to `10`, [sklearn's default value](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html).
- verbose: Wheter to be verbose or not. Defaults to `False`.

### fit
```python
def fit(self, dataset: BinaryLabelDataset, preprocessed_datasets=None)
```
- dataset: The training dataset.
- preprocessed_datasets: This argument can be used in evaluation pipelines to make runs more efficient. If a dataset is specified, Fairboost will skip the steps that generates the datasets for the models (preprocessing + bootstrapping). Is usually obtained from a previous run of Fairboost. Defaults to `None`. 

### predict
```python
def predict(self, dataset: BinaryLabelDataset) -> np.array
```
- dataset: The test dataset.

## Trivia

### Wrappers
Some of AIF360 preprocessing techniques require slightly different preprocessing of the data before being transformed by the unfairness mitigation techniques. However, Fairboost receives only one dataset in the `fit` function, which means only one version of the data will be sent to the preprocessing techniques. To overcome this challenge, we suggest using wrapper classes, has it has been done in the `wrappers.py` file. 

This strategy can also be used to use preprocessing techniques of other libraries in Fairboost.

