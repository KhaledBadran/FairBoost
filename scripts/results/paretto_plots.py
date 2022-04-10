import json
from typing import List, Dict

import pandas as pd
import numpy as np
from pathlib import Path
from typeguard import typechecked
import enum
import seaborn as sns
import matplotlib.pyplot as plt


# TODO: this is probably already defined somehwere


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


def read_results(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def read_data() -> Dict:
    # Building paths
    file_dir = Path(__file__).parent.resolve()
    data_path = Path(file_dir, "raw_data").resolve()
    fairboost_results_path = Path(data_path, 'fairboost_splits.json')
    baseline_results_path = Path(data_path, 'baseline_splits.json')

    data = {}
    data['baseline'] = read_results(baseline_results_path)['results']
    data['fairboost'] = read_results(fairboost_results_path)['results']
    return data


def results_to_dataframe(results, preprocessing_method):
    # Getting the length of arrays (the number of seeds)
    len_ = len(results[list(results.keys())[0]])
    results['preprocessing method'] = [
        str(preprocessing_method) for _ in range(len_)]
    results['fairness'] = 1 - \
        np.abs((1-np.array(results['disparate_impact'])))
    return pd.DataFrame.from_dict(results)


def get_fairboost_datapoints(data: Dict, dataset_name, classifier_name, boostrap_type: Bootstrap_type):
    all_types = set([str(Preprocessing_names.LFR), str(
        Preprocessing_names.OptimPreproc), str(Preprocessing_names.Reweighing)])
    for config in data['fairboost'][dataset_name]['fairboost']:
        # If it is the version of Fairboost with all preprocessing techniques
        if set(config['hyperparameters']['preprocessing'].keys()) == all_types:
            # If it is the correct bootstrap type
            if config['hyperparameters']['init']['bootstrap_type'] == boostrap_type:
                results = config['results'][classifier_name]
    return results_to_dataframe(results, f'Fairboost_{boostrap_type}')


def get_preproc_datapoints(data: Dict, dataset_name, classifier_name, preprocessing_method: Preprocessing_names):
    if preprocessing_method == Preprocessing_names.Baseline:
        results = data['baseline'][dataset_name][preprocessing_method][classifier_name]
    else:
        results = data['baseline'][dataset_name][preprocessing_method][0]['results'][classifier_name]
    return results_to_dataframe(results, preprocessing_method)


def to_dataframe(data: Dict, dataset_name, classifier_name):
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


def plot_paretto_front(df, dataset_name, classifier_name, print_figures=False, plots_dir=Path("plots/")):
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

    plots_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"paretto-{dataset_name}-{classifier_name}.pdf"
    file_path = Path(plots_dir, file_name)
    p.figure.savefig(file_path)


def pareto_frontier(ObjectiveX, objectiveY, optimizeObjectiveX=True, optimizeObjectiveY=True):
    '''
    #https://sirinnes.wordpress.com/2013/04/25/pareto-frontier-graphic-via-python/
    Method to take two equally-sized lists and return just the elements which lie 
    on the Pareto frontier, sorted into order.
    Default behaviour is to find the maximum for both X and Y, but the option is
    available to specify maxX = False or maxY = False to find the minimum for either
    or both of the parameters.
    '''
    orderedList = sorted([[ObjectiveX[i], objectiveY[i]]
                         for i in range(len(ObjectiveX))], reverse=optimizeObjectiveX)
    p_front = [orderedList[0]]

    for pair in orderedList[1:]:
        if optimizeObjectiveY:
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

    for dataset_name in dataset_names:
        for model_name in model_names:
            df = to_dataframe(data, dataset_name, model_name)
            plot_paretto_front(df, dataset_name, model_name)


if __name__ == '__main__':
    main()
