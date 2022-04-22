from utils import get_data_h_mean
from scipy.stats import ttest_ind
import pandas as pd


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


def get_t_value(f_results, e_results):
    return ttest_ind(f_results, e_results, equal_var=False, alternative='greater')


def get_t_values(df, dataset, classifier):
    b_results = get_baseline_results(df, dataset, classifier)

    res = []
    for bootstrap_type in ['none', 'default', 'custom']:
        e_results = get_ensemble_baseline_results(
            df, dataset, classifier, bootstrap_type)
        f_results = get_fairboost_result(
            df, dataset, classifier, bootstrap_type)
        res.append({'classifier': classifier, 'dataset': dataset, 'bootstrap_type': bootstrap_type,
                    't_val_vs_baseline': get_t_value(f_results, b_results), 't_val_vs_ensemble': get_t_value(f_results, e_results)})
    return res


def main():
    res = []
    df = get_data_h_mean()
    for dataset in ['german', 'adult', 'compas']:
        for classifier in ['Logistic Regression', 'Random Forest']:
            res = [*res, *get_t_values(df, dataset, classifier)]
    res = pd.DataFrame(res)
    print(res)


if __name__ == "__main__":
    main()
