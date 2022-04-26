import json
from typing import List, Dict, Tuple
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typeguard import typechecked
import enum
import seaborn as sns
import matplotlib.pyplot as plt

from utils import get_plots_dir


class Preprocessing_names(str, enum.Enum):
    def __str__(self):
        return str(self.value)

    LFR = 'LFR'
    DIR = 'DisparateImpactRemover'
    OptimPreproc = 'OptimPreproc'
    Reweighing = 'Reweighing'
    Baseline = 'baseline'


class Bootstrap_type(str, enum.Enum):
    NONE = "NONE"
    DEFAULT = 'DEFAULT'
    CUSTOM = 'CUSTOM'


@typechecked
def read_results(path: Path) -> Dict:
    """
    Reads a json file and returns a dict
            Parameters:
                    path : path of the json file.
            Returns:
                    data: the data in the file
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return data


@typechecked
def read_data() -> Dict:
    """
    Reads all the raw results files and return a dictionary
    with their information.
            Returns:
                    data: the data in the files
    """
    # Building paths
    file_dir = Path(__file__).parent.resolve()
    data_path = Path(file_dir, "raw_data").resolve()
    fairboost_results_path = Path(data_path, 'fairboost_splits.json')
    baseline_results_path = Path(data_path, 'baseline_splits.json')

    data = {}
    data['baseline'] = read_results(baseline_results_path)['results']
    data['fairboost'] = read_results(fairboost_results_path)['results']
    return data


@typechecked
def results_to_dataframe(results: Dict, preprocessing_method: str) -> pd.DataFrame:
    """
    Transforms a leaf-level dictionary of the result files into a Dataframe.
    Saves in the dataframe the preprocessing method name and the normalized
    disparate impact.
            Parameters:
                    results : The leaf-level dictionary with results.
                    preprocessing_method : name of the preprocessing method.
            Returns:
                    data: the dataframe version of the dictionary
    """
    # Getting the length of arrays (the number of seeds)
    len_ = len(results[list(results.keys())[0]])
    results['preprocessing method'] = [
        str(preprocessing_method) for _ in range(len_)]
    # Computing normalized disparate impact.
    results['fairness'] = [
        score if score <= 1 else (score**-1) for score in results['disparate_impact']
    ]
    return pd.DataFrame.from_dict(results)


@typechecked
def get_fairboost_datapoints(data: Dict, dataset_name: str, classifier_name: str, boostrap_type: Bootstrap_type) -> pd.DataFrame:
    """
    Fetches Fairboost's datapoints from the data dictionary
    with their information.
            Parameters:
                    data : The results of the experiments.
                    dataset_name: 
                    classifier_name: 
                    boostrap_type: The bootstrap type hyperparam of Fairboost.
            Returns:
                    Fairboost's datapoints
    """
    all_types = set([str(Preprocessing_names.LFR), str(
        Preprocessing_names.OptimPreproc), str(Preprocessing_names.Reweighing)])
    for config in data['fairboost'][dataset_name]['fairboost']:
        # If it is the version of Fairboost with all preprocessing techniques
        if set(config['hyperparameters']['preprocessing'].keys()) == all_types:
            # If it is the correct bootstrap type
            if config['hyperparameters']['init']['bootstrap_type'] == boostrap_type:
                results = config['results'][classifier_name]
    return results_to_dataframe(results, f'Fairboost_{boostrap_type}')


@typechecked
def get_preproc_datapoints(data: Dict, dataset_name: str, classifier_name: str, preprocessing_method: Preprocessing_names) -> pd.DataFrame:
    """
    Fetches Fairboost's datapoints from the data dictionary
    with their information.
            Parameters:
                    data : The results of the experiments.
                    dataset_name: 
                    classifier_name: 
                    preprocessing_method: The name of the preprocessing method.
            Returns:
                    Preprocessing methods' datapoints
    """
    if preprocessing_method == Preprocessing_names.Baseline:
        results = data['baseline'][dataset_name][preprocessing_method][classifier_name]
    else:
        results = data['baseline'][dataset_name][preprocessing_method][0]['results'][classifier_name]
    return results_to_dataframe(results, str(preprocessing_method))


@typechecked
def to_dataframe(data: Dict, dataset_name: str, classifier_name: str) -> pd.DataFrame:
    """
    Transform the dictionary with the results into a dataframe.
            Parameters:
                    data : The results of the experiments.
                    dataset_name: 
                    classifier_name: 
            Returns:
                    data: The results of the experiments.
    """
    dfs = []
    # Fetching data points for each preprocessing method
    for preprocessing_method in Preprocessing_names:
        # Trying to be agnostic of the actual metric there
        dfs.append(get_preproc_datapoints(
            data, dataset_name, classifier_name, preprocessing_method))
    # Fetching data points for each fairboost configuration
    for fairboost_version in Bootstrap_type:
        dfs.append(get_fairboost_datapoints(
            data, dataset_name, classifier_name, fairboost_version))
    return pd.concat(dfs)


@typechecked
def plot_paretto_front(df: pd.DataFrame, dataset_name: str, classifier_name: str, print_figures=False, plots_dir=Path("plots/")):
    """
    Plots the paretto front.
            Parameters:
                    df : The results of the experiments.
                    dataset_name: 
                    classifier_name: 
                    print_figures: True to show figures.
                    plots_dir: Where to save the plots.
    """
    X_axis_metric, Y_axis_metric = 'f1-score', 'fairness'
    sns.set_theme(style="white")
    p = sns.relplot(x=X_axis_metric, y=Y_axis_metric,
                    hue="preprocessing method", data=df)
    p.fig.suptitle(
        f'Dataset: {dataset_name} - Model: {classifier_name}')

    Xs = df[X_axis_metric].to_numpy()
    Ys = df[Y_axis_metric].to_numpy()
    p_front = pareto_frontier(Xs, Ys)
    plt.plot(p_front[0], p_front[1], 'k--')
    if print_figures:
        plt.show()

    file_name = f"{dataset_name}-{classifier_name}.pdf"
    file_path = plots_dir/file_name
    p.figure.savefig(str(file_path))


@typechecked
def pareto_frontier(X: np.array, Y: np.array, optimizeX=True, optimizeY=True) -> Tuple:
    '''
    Fetches the points of the paretto frontier.
    Taken from: https://sirinnes.wordpress.com/2013/04/25/pareto-frontier-graphic-via-python/
            Parameters:
                    X: The X values of the datapoint in the graph.
                    Y: The Y values of the datapoint in the graph.
                    optimizeX: Whether to maximize or not on the X axis.
                    optimizeY: Whether to maximize or not on the Y axis.
            Returns:
                    (p_frontX, p_frontY): The X and Y coordinates of the points of the paretto frontier.
    '''
    orderedList = sorted([[X[i], Y[i]]
                         for i in range(len(X))], reverse=optimizeX)
    p_front = [orderedList[0]]

    for pair in orderedList[1:]:
        if optimizeY:
            if pair[1] >= p_front[-1][1]:
                p_front.append(pair)
        else:
            if pair[1] <= p_front[-1][1]:
                p_front.append(pair)

    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]

    return p_frontX, p_frontY

def main():
    data = read_data()
    dataset_names = ['german', 'adult', 'compas']
    model_names = ['Logistic Regression', 'Random Forest']
    plots_dir = get_plots_dir('paretto_plots')

    for dataset_name in dataset_names:
        for model_name in model_names:
            df = to_dataframe(data, dataset_name, model_name)
            plot_paretto_front(df, dataset_name, model_name, plots_dir=plots_dir)


if __name__ == '__main__':
    main()
