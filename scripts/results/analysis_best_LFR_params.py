import pandas as pd
import numpy as np
from pathlib import Path
from typeguard import typechecked
import json
from typing import List, Dict
from statistics import harmonic_mean, mean
import os

@typechecked
def read_data_LFR_evaluation(path: Path) -> List[Dict]:
    """
    Reads the results of evaluating LFR's best parameters and returns a list of dicts representing the evaluation results
            Parameters:
                    path : path of the file LFR_evaluation.json
            Returns:
                    LFR_results: a list where each item is a dictionary containing the results from
                    one experiment run such as (German - LFR - Random Forrest - Hyperparams(X) )
    """
    LFR_results = []

    with open(path, "r") as f:
        results = json.load(f)

    for dataset, dataset_results in results["results"].items():

        raw_results = dataset_results['LFR']

        for raw_result in raw_results:

            for cls, metrics in raw_result['results'].items():
                LFR_results.append(
                    {
                        "dataset": dataset,
                        "params": raw_result['hyperparameters'],
                        "classifier": cls,
                        "metrics": metrics,
                    }
                )
    return LFR_results


def calculate_h_mean(f1_list: List, di_list: List) -> List:
    """
    returns a list of harmonic means based on each (f1, di) tuples
    :param f1_list: list of f1 scores
    :param di_list: list of normalized disparate impact scores
    :return: list of harmonic mean scores
    """
    return list(
        map(lambda x, y: harmonic_mean([x, y]), f1_list, di_list)
    )


def save_best_params(df):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    results_dir = Path(dir_path, "LFR_best_params")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = Path(results_dir, "LFR_best_params.csv")
    df.to_csv(results_file)

def main():

    data_path = Path("raw_data")
    LFR_results_path = Path(data_path, "LFR_evaluation.json")
    data = read_data_LFR_evaluation(LFR_results_path)
    df = pd.DataFrame(data)

    # explode metrics into separate columns
    df = pd.concat([df.drop(['metrics'], axis=1), df['metrics'].apply(pd.Series)], axis=1)

    # calculate normalized disparate impact
    df['n_disparate_impact'] = df['disparate_impact'].apply(lambda di_list: [di if di <= 1 else (di ** -1) for di in di_list])

    # calculate the list of harmonic means based on the list of f1 scores and normalized DI
    df['h_mean'] = df.apply(lambda row: calculate_h_mean(row['f1-score'], row['n_disparate_impact']), axis=1)

    # average all h_means
    df['avg_h_mean'] = df['h_mean'].apply(lambda x: mean(x))

    # keep only the columns of intrest
    df = df[['dataset', 'params', 'classifier', 'avg_h_mean']]

    # represent params as a string to be used in the groupby
    df['params'] = df['params'].apply(lambda x: str(x))

    # aggregate the results by calculating the mean from both classifiers
    df = df.groupby(by=['dataset', 'params']).mean().reset_index()

    # find the best params for each dataset
    df = df.sort_values('avg_h_mean', ascending=False).groupby(['dataset'], sort=False).head(1)

    # save results
    save_best_params(df)

if __name__ == "__main__":
    main()
