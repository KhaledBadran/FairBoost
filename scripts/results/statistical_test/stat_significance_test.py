import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu
from collections import defaultdict
import json


def generate_ensemble_name(r, unique_name):
    ensemble_name = r['method'].split('+')
    if len(ensemble_name) > 2:
        return 'All'
    else:
        ensemble_name.remove(unique_name)
        return f"+{ensemble_name[0]}"


def to_array(arr_str):
    res = arr_str[1:-1].split(', ')
    res = map(float, res)
    return list(res)


def convert_res_to_array(row):
    row['accuracy'] = to_array(row['accuracy'])
    row['fairness'] = to_array(row['fairness'])
    return row

def is_significant(p_value, alpha):
    return 'Significant' if p_value < alpha else 'Not significant'

def main():
    alpha = 0.05

    crnt_dir = crnt_dir = Path(__file__).resolve().parent
    data_file = crnt_dir/'statistical_tests_raw_data.csv'
    # Load data
    df = pd.read_csv(data_file)
    df = df.astype({'accuracy': 'object', 'fairness': 'object'})

    def nested_dict(): return defaultdict(nested_dict)
    res = nested_dict()

    for classifier in ['Random Forest', 'Logistic Regression']:
        for dataset in ['german', 'compas', 'adult']:
            # Comparing ensembles against unique techniques
            for unique_technique in ['LFR', 'OP', 'RW']:
                # Filter out irrelevant columns given the current iteration
                df_filtered = df[(df['dataset'] == dataset) & (
                    df['classifier'] == classifier) & (df['method'].str.contains(unique_technique))]
                # Get unique & ensemble results
                unique_results = df_filtered[df['method']
                                             == unique_technique].squeeze(axis=0)
                unique_results = convert_res_to_array(unique_results)
                ensemble_results = df_filtered[df['method']
                                               != unique_technique]

                for i, r in ensemble_results.iterrows():
                    r = convert_res_to_array(r)

                    # Run mannwhitney test for accuracy results
                    _, acc_p_value = mannwhitneyu(
                        r['accuracy'], unique_results['accuracy'])
                    # Run mannwhitney test for fairness results
                    _, fairness_p_value = mannwhitneyu(
                        r['fairness'], unique_results['fairness'])
                    # Save results in dict
                    ensemble_name = generate_ensemble_name(
                        r, unique_results['method'])
                    res[classifier][dataset][unique_technique][ensemble_name]['accuracy'] = is_significant(acc_p_value, alpha)
                    res[classifier][dataset][unique_technique][ensemble_name]['fairness'] = is_significant(fairness_p_value, alpha)

            # Comparing unique techniques against baseline
            df_filtered = df[(df['dataset'] == dataset) &
                             (df['classifier'] == classifier)]
            baseline = df_filtered[df['method'] == 'Baseline'].squeeze(axis=0)
            baseline = convert_res_to_array(baseline)
            for unique_technique in ['LFR', 'OP', 'RW']:
                unique_results = df_filtered[df['method']
                                             == unique_technique].squeeze(axis=0)
                unique_results = convert_res_to_array(
                    unique_results).squeeze(axis=0)
                # Run mannwhitney test for accuracy results
                _, acc_p_value = mannwhitneyu(
                    baseline['accuracy'], unique_results['accuracy'])
                # Run mannwhitney test for fairness results
                _, fairness_p_value = mannwhitneyu(
                    baseline['fairness'], unique_results['fairness'])
                res[classifier][dataset]['Baseline'][unique_technique]['accuracy'] = is_significant(acc_p_value, alpha)
                res[classifier][dataset]['Baseline'][unique_technique]['fairness'] = is_significant(fairness_p_value, alpha)

    res_for_df = []
    for classifier in ['Random Forest', 'Logistic Regression']:
        for dataset in res[classifier].keys():
            for method_1 in res[classifier][dataset].keys():
                for method_2 in res[classifier][dataset][method_1].keys():
                    res_for_df.append({
                        'classifier': classifier,
                        'dataset': dataset,
                        'method_1': method_1,
                        'method_2': method_2,
                        'accuracy': res[classifier][dataset][method_1][method_2]['accuracy'],
                        'fairness': res[classifier][dataset][method_1][method_2]['fairness']
                    })

    print(res_for_df)
    df = pd.DataFrame(res_for_df)
    # escape + sign for csv
    df['method_1'] = df['method_1'].str.replace('+', '\+')
    df['method_2'] = df['method_2'].str.replace('+', '\+')
    # df = df.replace('+', "\+", regex=True)
    df.to_csv(crnt_dir/'statistical_tests_results.csv', index=False)

    with open(crnt_dir/'results.json', 'w') as f:
        f.write(json.dumps(res, indent=4))


if __name__ == '__main__':
    main()
