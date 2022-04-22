# FairBoost
Fairboost is a unfairness tools that combines preprocessing tools via an ensemble to achieve fair and accurate results. For more details, refer to the paper.  

## Installation
Because of AIF360, some dependencies have to be manually handled. We recommend replicating the following procedure for the installation of our project. It will suppose you have conda installed.

1. Start by creating a virtual environment with python 3.7 installed
```Shell
conda create --name fairboost python=3.7
``` 

2. Activate the environment
```Shell
conda activate fairboost
``` 

3. Pre-install a dependency of AIF360 
```Shell
conda install -c conda-forge cvxpy
```

4. Install the project's dependencies
```Shell
pip install -r requirements.txt
```

5. Download AIF360 datasets and install them at the correct path in AIF360 library. The library has not automated yet this process. You can refer to their the README in [this folder of their Github repo](https://github.com/Trusted-AI/AIF360/tree/master/aif360/data).


## How to use fairboost
Fairboost's API is inspired by sklearn. Using it should feel familiar. To use Fairboost, we must first define the preprocessing algorithms we will use:

```python
from aif360.algorithms.preprocessing import Reweighing, OptimPreproc, LFR, DisparateImpactRemover

pp1 = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
pp2 = DisparateImpactRemover(repair_level=.5)
pp = [pp1, pp2]
```
For this exemple, we use only Reweighing and DisparateImpactRemover. The complete list of preprocessing techniques for unfairness mitigation can be found on [AIF360 docs](https://aif360.readthedocs.io/en/latest/modules/algorithms.html#module-aif360.algorithms.preprocessing).

We select the model that will be used by Fairboost.

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
```

Finally, we instantiate Fairboost using the model and the preprocessing techniques we have defined before. Fairboost is trained via `fit` function and output predictions via the `predict` function.

```python
ens = FairBoost(model, pp, bootstrap_type=Bootstrap_type.CUSTOM)
ens = ens.fit(dataset_orig_train)
y_pred = ens.predict(dataset_orig_test)
accuracy_score(y_pred, dataset_orig_test.labels)
```
For more details, consult the [README in the Fairboost folder](scripts/FairBoost/README.md).


## Directory structure

Below, the directory structure of the project with the most important files shown.

```
FairBoost/
│   README.md
│   requirements.txt    
│
└───scripts/
│   │   demo_
│   │   evaluation_
│   │   find_LFR_best_params.py
│   │
│   └───configs/
│   │   │   constants.py
│   │   
│   │   
│   └───FairBoost/
│   │   │   main.py
│   │   │   README.md
│   │
│   │
│   └───results/
│       │   paretto_plots.py
│       │   rectangular_plot.py
│       └───plots/
│       └───raw_data/
```

### Folder: script
Most of the code is under the `script/` folder. In it, we can find files named with these prefixes:
- `demo_`: they serve as an exemple on how to use Fairboost.
- `evaluation_`: files that generated the results we analyzed for our paper.

The `find_LFR_best_params.py` file was used to find working hyperparameter configurations for LFR for each dataset.

### Folder: configs
Contains the hyperparameter configurations used to run the evaluation pipelines. To import them, import the `constants.py` file.

### Folder: Fairboost
Contains Fairboost's code. To import Fairboost, import `main.py` file.

### Folder: results
Folder with the output of the evaluation pipeline. The raw results can be found under `raw_data` directory. The files suffixed with `_plot` plot the data from `raw_data/` and save the resulting figures under `plots/`. 



