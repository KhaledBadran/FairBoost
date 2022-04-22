from typing import List, Tuple

from typeguard import typechecked
from utils import get_data_h_mean
from scipy.stats import ttest_ind
import pandas as pd
import os
from pathlib import Path


@typechecked
def get_fairboost_result(df: pd.DataFrame, dataset: str, classifier: str, bootstrap_type: str) -> List:
    """
    Helper function to fetch data from a specific dataframe.
    Returns the h_mean results from a fairboost experiment. 
        Parameters:
            df : dataframe from which data must be extracted
            dataset: The name of the dataset on which the experiment was run.
            classifier: The name of the classifier used for the experiment.
            bootstrap_type: The bootstrapping type (NONE, DEFAULT, CUSTOM)
        Returns:
            results (list): the results for a specific experiment
    """
    row = df[(df['experiment'] == 'fairboost') &
             (df['preprocessing'] == 'LFR,OptimPreproc,Reweighing') &
             (df['dataset'] == dataset) &
             (df['classifier'] == classifier) &
             (df['bootstrap_type'] == bootstrap_type)
             ]
    return row['h_mean'].tolist()[0]


@typechecked
def get_baseline_results(df: pd.DataFrame, dataset: str, classifier: str) -> List:
    """
    Helper function to fetch data from a specific dataframe.
    Returns the h_mean results from a baseline experiment. 
        Parameters:
            df : dataframe from which data must be extracted
            dataset: The name of the dataset on which the experiment was run.
            classifier: The name of the classifier used for the experiment.
        Returns:
            results (list): the results for a specific experiment
    """
    row = df[(df['experiment'] == 'baseline') &
             (df['dataset'] == dataset) &
             (df['classifier'] == classifier)
             ]
    return row['h_mean'].tolist()[0]


@typechecked
def get_ensemble_baseline_results(df: pd.DataFrame, dataset: str, classifier: str, bootstrap_type: str) -> List:
    """
    Helper function to fetch data from a specific dataframe.
    Returns the h_mean results from an "ensemble baseline" experiment. 
        Parameters:
            df : dataframe from which data must be extracted
            dataset: The name of the dataset on which the experiment was run.
            classifier: The name of the classifier used for the experiment.
            bootstrap_type: The bootstrapping type (NONE, DEFAULT, CUSTOM)
        Returns:
            results (list): the results for a specific experiment
    """
    row = df[(df['experiment'] == 'ensemble') &
             (df['dataset'] == dataset) &
             (df['classifier'] == classifier) &
             (df['bootstrap_type'] == bootstrap_type)
             ]
    return row['h_mean'].tolist()[0]


@typechecked
def run_t_test(f_results: List, b_results: List) -> Tuple:
    """
    Wrapper around scipy ttest function. It tests whether Fairboost
    has significant better results the given baseline results.

    Runs ttests to compare fairboost results against other ones. 
    The test is always one-sided and has the alternative
    hypothesis that Fairboost has better results.
    Parameters:
            f_results : Fairboost results
            b_results: A baseline results
        Returns:
            results: the t_value and p_value resulting of the test
    """
    return ttest_ind(f_results, b_results, equal_var=False, alternative='greater')


@typechecked
def get_t_test_results(df: pd.DataFrame, dataset: str, classifier: str) -> Tuple[List, List]:
    """
    Compare Faiboost results against baselines (simple and ensemble)
    using t-tests. 
        Parameters:
            df : dataframe with all the results of an experiment
            dataset: The name of the dataset for which we want to analyze results.
            classifier: The name of the classifier for which we want to analyze results.
        Returns:
            results: The t-values and p-values of the experiments 
    """
    b_results = get_baseline_results(df, dataset, classifier)

    t_values, p_values = [], []
    for bootstrap_type in ['none', 'default', 'custom']:
        # Fetching results from the dataframe
        e_results = get_ensemble_baseline_results(
            df, dataset, classifier, bootstrap_type)
        f_results = get_fairboost_result(
            df, dataset, classifier, bootstrap_type)

        # Running t_test. Comparing against both baselines (baseline & ensemble)
        e_t_value, e_p_value = run_t_test(f_results, e_results)
        b_t_value, b_p_value = run_t_test(f_results, b_results)

        # Formatting the results so it can be transformed in a Dataframe later on.
        general_info = {'classifier': classifier,
                        'dataset': dataset, 'bootstrap_type': bootstrap_type}
        t_values.append(
            {**general_info, 't_val_vs_baseline': b_t_value, 't_val_vs_ensemble': e_t_value})
        p_values.append(
            {**general_info, 't_val_vs_baseline': b_p_value, 't_val_vs_ensemble': e_p_value})
    return (t_values, p_values)


@typechecked
def save_results(t_values: pd.DataFrame, p_values: pd.DataFrame):
    """
    Saves t_values and p_values in a csv format.
        Parameters:
            t_values : The t-values comparing experiments
            p_values: The t-values comparing experiments
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    results_dir = Path(dir_path, "significance_tests")
    results_dir.mkdir(parents=True, exist_ok=True)
    t_values.to_csv(Path(results_dir, 'h_mean_t_values.csv'))
    p_values.to_csv(Path(results_dir, 'h_mean_p_values.csv'))


def main():
    t_values, p_values = [], []
    df = get_data_h_mean()
    for dataset in ['german', 'adult', 'compas']:
        for classifier in ['Logistic Regression', 'Random Forest']:
            t, p = get_t_test_results(df, dataset, classifier)
            t_values, p_values = [*t_values, *t], [*p_values, *p]
    t_values, p_values = pd.DataFrame(t_values), pd.DataFrame(p_values)
    save_results(t_values, p_values)


if __name__ == "__main__":
    main()
