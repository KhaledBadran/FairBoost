from utils import get_data_h_mean
from scipy.stats import ttest_ind
import pandas as pd
import os
from pathlib import Path


def get_fairboost_result(df, dataset, classifier, bootstrap_type):
    """
    Fetches fairboost results from the dataframe, 
    given the dataset and the classifier. Always
    fetches 

    """
    row = df[(df['experiment'] == 'fairboost') &
             (df['preprocessing'] == 'LFR,OptimPreproc,Reweighing') &
             (df['dataset'] == dataset) &
             (df['classifier'] == classifier) &
             (df['bootstrap_type'] == bootstrap_type)
             ]
    return row['h_mean'].tolist()[0]


def get_baseline_results(df, dataset, classifier):
    row = df[(df['experiment'] == 'baseline') &
             (df['dataset'] == dataset) &
             (df['classifier'] == classifier)
             ]
    return row['h_mean'].tolist()[0]


def get_ensemble_baseline_results(df, dataset, classifier, bootstrap_type):
    row = df[(df['experiment'] == 'ensemble') &
             (df['dataset'] == dataset) &
             (df['classifier'] == classifier) &
             (df['bootstrap_type'] == bootstrap_type)
             ]
    return row['h_mean'].tolist()[0]


def run_t_test(f_results, e_results):
    return ttest_ind(f_results, e_results, equal_var=False, alternative='greater')


def get_t_test_results(df, dataset, classifier):
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
    return t_values, p_values


def save_results(t_values: pd.DataFrame, p_values: pd.DataFrame):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    results_dir = Path(dir_path, "significance_tests")
    results_dir.mkdir(parents=True, exist_ok=True)
    t_values.to_csv(Path(results_dir, 'h_mean_t_values'))
    p_values.to_csv(Path(results_dir, 'h_mean_p_values'))


def main():
    t_values, p_values = [], []
    df = get_data_h_mean()
    for dataset in ['german', 'adult', 'compas']:
        for classifier in ['Logistic Regression', 'Random Forest']:
            results = get_t_test_results(df, dataset, classifier)
            t_values, p_values = [*t_values, *
                                  results[0]], [*p_values, *results[1]]
    t_values, p_values = pd.DataFrame(t_values), pd.DataFrame(p_values)
    save_results(t_values, p_values)


if __name__ == "__main__":
    main()
