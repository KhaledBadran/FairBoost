import pandas as pd
import numpy as np
from pathlib import Path

from typeguard import typechecked
import json
from typing import List, Dict
from statistics import harmonic_mean, mean
import seaborn as sns
import matplotlib.pyplot as plt


@typechecked
def read_data_baseline(path: Path) -> List[Dict]:
    """
    Generates a list of dicts representing the data in the baseline_splits.json.
            Parameters:
                    path : path of the file baseline_splits.json
            Returns:
                    baseline_results: a list where each item is a dictionary representing the containg the results from
                    one experiment run such as (preprocessing - LFR - Random Forrest)
    """
    baseline_results = []
    with open(path, "r") as f:
        results = json.load(f)

    for dataset, dataset_results in results["results"].items():
        for (
                preprocessing_method,
                preprocessing_method_results,
        ) in dataset_results.items():

            # Baseline experiment without any preprocessing
            if preprocessing_method == "baseline":
                for (
                        classifier,
                        performance_metrics,
                ) in preprocessing_method_results.items():
                    baseline_results.append(
                        {
                            "experiment": "baseline",
                            "bootstrap_type": "No bootstrap",
                            "dataset": dataset,
                            "preprocessing": ["No Preprocessing"],
                            "classifier": classifier,
                            "metrics": performance_metrics,
                        }
                    )

            # Experiments with preprocessing only (e.g., LFR, Reweighing, OptimPrepr)
            elif preprocessing_method != "DisparateImpactRemover":
                for classifier, performance_metrics in preprocessing_method_results[0][
                    "results"
                ].items():
                    baseline_results.append(
                        {
                            "experiment": "preprocessing",
                            "bootstrap_type": "No bootstrap",
                            "dataset": dataset,
                            "preprocessing": [preprocessing_method],
                            "classifier": classifier,
                            "metrics": performance_metrics,
                        }
                    )
    return baseline_results


@typechecked
def get_fairboost_run_results(
        dataset_name: str, raw_run_results: Dict, with_preprocessing: bool
) -> List[Dict]:
    """
    Analyzes the raw json results of a specific fairboost run and returns the results as a proper list of dictionaries.
    :param dataset_name: name of the dataset used
    :param raw_run_results: the json/dict that contains the information about the run (e.g., which bootstrap method was
     used)
    :param with_preprocessing: whether the results are from a run with ensemble only (no preprocessing) or
    whether it includes preprocessing (fairboost)
    :return: a list of dictionaries containing the information about the experiments form the raw run results.
    """
    run_results = []

    if with_preprocessing:
        # this is the fairboost results (ensemble + preprocessing)
        experiment = "fairboost"

        # get the preprocessing methods used in the run (e.g., [LFR, OptimPreproc] or [Reweighing])
        preprocessing_methods = list(
            raw_run_results["hyperparameters"]["preprocessing"].keys()
        )

        # get the bootstrap method used by the ensemble (None, Default, or Custom)
        bootstrap_method = raw_run_results["hyperparameters"]["init"][
            "bootstrap_type"
        ].lower()

    else:
        # since this is a fairboost baseline/normal ensemble we don't have preprocessing
        experiment = "ensemble"
        preprocessing_methods = ["No Preprocessing"]

        # get the bootstrap method used by the ensemble (None, Default, or Custom)
        bootstrap_method = raw_run_results["hyperparameters"]["bootstrap_type"].lower()

    # iterate over classifiers to get their performance metrics
    for classifier, performance_metrics in raw_run_results["results"].items():
        run_results.append(
            {
                "experiment": experiment,
                "bootstrap_type": bootstrap_method,
                "dataset": dataset_name,
                "preprocessing": preprocessing_methods,
                "classifier": classifier,
                "metrics": performance_metrics,
            }
        )

    return run_results


def read_data_fairboost(path):
    """
    Generates a list of dicts representing the data in the fairboost_splits.json. This includes the results for the
    ensemble only (without preprocessing) and fairboost (ensemble + preprocessing).
            Parameters:
                    path : path of the file fairboost_splits.json
            Returns:
                    all_results: a list where each item is a dictionary representing the results from one experiment
                     run such as (Fairboost - german - default - [LFR, OptimPreproc] - Random Forrest)
    """
    all_results = []
    with open(path, "r") as f:
        results = json.load(f)

    for dataset, dataset_results in results["results"].items():

        # results when ensemble doesn't apply preprocessing techniques
        ensemble_only_results = dataset_results["baseline"]

        for run in ensemble_only_results:
            ensemble_run_results = get_fairboost_run_results(
                dataset_name=dataset, raw_run_results=run, with_preprocessing=False
            )
            all_results.extend(ensemble_run_results)

        # results for fairboost (ensemble + preprocessing)
        fairboost_results = dataset_results["fairboost"]

        for run in fairboost_results:
            fairboost_run_results = get_fairboost_run_results(
                dataset_name=dataset, raw_run_results=run, with_preprocessing=True
            )
            all_results.extend(fairboost_run_results)

    return all_results


@typechecked
def read_data() -> List[Dict]:
    """
    Read data from files and return its content as a list of dictionnaries.
            Returns:
                    data: the data contained in both files
    """
    data_path = Path("raw_data")
    fairboost_results_path = Path(data_path, "fairboost_splits.json")
    baseline_results_path = Path(data_path, "baseline_splits.json")
    data_baseline = read_data_baseline(baseline_results_path)
    data_fairboost = read_data_fairboost(fairboost_results_path)
    return data_baseline + data_fairboost


@typechecked
def compute_h_mean(f1_scores: List, normalized_di_scores: List):
    """
    this method calculates the harmonic mean for each (f1-score, normalized_di_score) tuple.
    :param f1_scores: list of f1_score for each of the 10 train-test split runs
    :param normalized_di_scores: list of normalized (between 0 and 1) disparate impact scores for each of the 10
     train-test split runs
    :return: a list of harmonic means for each run
    """
    harmonic_means = list(
        map(lambda x, y: harmonic_mean([x, y]), f1_scores, normalized_di_scores)
    )
    return harmonic_means


@typechecked
def list_to_string(l: List[str]) -> str:
    """
    changes a list into a string (e.g., ['LFR', 'Reweighing'] --> 'LFR, Reweighing'
    :param l: list of strings
    :return: the list item concatenated into strings
    """
    return ",".join(map(str, l))


@typechecked
def preprocess_dataframe(df, dataset: str, classifier: str, n_elem=5):
    """
    Preprocesses the dataframe as required by the seaborn library.
    :param df: dataframe to be preprocessed
    :param dataset: the dataset name
    :param classifier: the classifier name
    :param n_elem: the number of configuration to analyse (e.g 5 means the top 5 configs)
    :return: the preprocessed dataframe
    """
    # select the rows by the classifiers and datasets names
    preprecessed_df = df.loc[(df["dataset"] == dataset) & (df["classifier"] == classifier)]

    # select the top n_elems having the highest mean of the h_mean
    preprecessed_df['Mean'] = preprecessed_df["h_mean"].apply(np.median)
    preprecessed_df = preprecessed_df.sort_values("Mean", ascending=False)[:n_elem]

    # Explode the h_mean list to rows
    preprecessed_df = preprecessed_df.explode("h_mean")

    preprecessed_df["preprocessing"] = preprecessed_df["preprocessing"].str.replace("Reweighing", "RW")
    preprecessed_df["preprocessing"] = preprecessed_df["preprocessing"].str.replace("OptimPreproc", "OP")

    # Add the column (1) experiment/(2) bootstrap_type/ (3)preprocessing which will be used as the x_axis
    preprecessed_df["(1) experiment / (2) bootstrap_type / (3)preprocessing"] = \
        "(1) " + preprecessed_df["experiment"].str.upper() + "\n" \
        + "(2) " + preprecessed_df["bootstrap_type"].str.upper() + "\n" \
        + "(3) " + preprecessed_df["preprocessing"].str.upper()

    return preprecessed_df


def plot_unique_boxplot(df):
    """
    this method will generate a plot for every combination of classifiers and datasets.
    The plots generated will have the name {classifier}_{dataset}.pdf
    :param df: the dataframe of the data
    """
    classifiers_list = df["classifier"].unique()
    datasets_list = df["dataset"].unique()
    for classifier in classifiers_list:
        for dataset in datasets_list:
            plot_df = preprocess_dataframe(df=df, dataset=dataset, classifier=classifier, n_elem=5)

            plot_title = "h_mean distribution of " + classifier + " model \n trained on the dataset " + dataset

            # sns.set(style="darkgrid")
            fig, ax = plt.subplots(figsize=(16, 13))
            # ax.set(ylim=(.5, 1))

            sns.set_context("paper", font_scale=2, rc={"font.size": 20, "axes.titlesize": 20, "axes.labelsize": 20})
            # sns.set(font_scale=1)

            PROPS = {
                'boxprops': {'facecolor': 'none', 'edgecolor': 'black'},
                'medianprops': {'color': 'black'},
                'whiskerprops': {'color': 'black'},
                'capprops': {'color': 'black'}
            }
            plot = sns.boxplot(x="(1) experiment / (2) bootstrap_type / (3)preprocessing", y="h_mean", data=plot_df,
                               ax=ax, showfliers=False, linewidth=3, width=0.5, **PROPS).set(title=plot_title)
            # plot.set_xlabel(fontsize=30)
            plt.savefig("Boxplots/" + classifier + "_" + dataset + ".pdf")


def plot_multiple_boxplots(df):
    """
    this method will generate a plot combining all the combinations of classifiers and datasets.
    The plots generated will have the name Combined_plots.pdf
    :param df: the dataframe of the data
    """
    classifiers_list = df["classifier"].unique()
    datasets_list = df["dataset"].unique()
    frames = []
    for classifier in classifiers_list:
        for dataset in datasets_list:
            plot_df = preprocess_dataframe(df=df, dataset=dataset, classifier=classifier, n_elem=5)
            frames.append(plot_df)

    results_df = pd.concat(frames)
    results_df = results_df.drop(['experiment', 'bootstrap_type', 'preprocessing', 'Mean'], axis=1)

    fig, ax = plt.subplots(figsize=(50, 50))

    PROPS = {
        'boxprops': {'facecolor': 'none', 'edgecolor': 'black'},
        'medianprops': {'color': 'black'},
        'whiskerprops': {'color': 'black'},
        'capprops': {'color': 'black'}
    }
    grid = sns.FacetGrid(results_df, col="classifier", row="dataset", height=4, aspect=2, sharey=True, sharex=False)
    grid.map(sns.boxplot, "(1) experiment / (2) bootstrap_type / (3)preprocessing", "h_mean", linewidth=1, width=0.5,
             **PROPS)

    plt.savefig("Boxplots/Combined_plots_classifier_as_rows_scaled_y.pdf")


def main():
    data = read_data()
    df = pd.DataFrame(data)
    df["preprocessing"] = [list_to_string(l) for l in df["preprocessing"]]

    # calculate avg harmonic mean and add it as a column
    h_mean_scores = []

    for performance_metric in df.loc[:, "metrics"]:
        f1_score = performance_metric["f1-score"]

        di_score = performance_metric["disparate_impact"]
        normalized_di_score = [
            score if score <= 1 else (score ** -1) for score in di_score
        ]

        h_mean_score = compute_h_mean(f1_score, normalized_di_score)
        h_mean_scores.append(h_mean_score)

    df["h_mean"] = h_mean_scores

    # drop the metrics column to apply aggregation afterwards
    df.drop(["metrics"], axis=1, inplace=True)

    # plot_unique_boxplot(df)
    plot_multiple_boxplots(df)


if __name__ == "__main__":
    main()
