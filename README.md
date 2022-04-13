# FairBoost


## Installation
Because of AIF360, some dependencies have to be manually handled. We recommend replicating the following procedure for the installation of our project. It will suppose you have conda installed.

1. Start by creating a virtual environment with python 3.7 installed
```bash
conda create --name fairboost python=3.7
``` 

2. Activate the environment
```bash
conda activate fairboost
``` 

3. Pre-install a dependency of AIF360 
``` bash
conda install -c conda-forge cvxpy
```

4. Install the project's dependencies
``` bash
pip install -r requirements.txt
```

5. Download AIF360 datasets and install them at the correct path in AIF360 library. The library has not automated yet this process. You can refer to their the README in [this folder of their Github repo](https://github.com/Trusted-AI/AIF360/tree/master/aif360/data).