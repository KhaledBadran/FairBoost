import pandas as pd
import numpy as np
from typeguard import typechecked
import seaborn as sns
import matplotlib.pyplot as plt

from utils import get_data_h_mean


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
    preprecessed_df = df.loc[(df["dataset"] == dataset)
                             & (df["classifier"] == classifier)]

    # select the top n_elems having the highest mean of the h_mean
    preprecessed_df['Mean'] = preprecessed_df["h_mean"].apply(np.median)
    preprecessed_df = preprecessed_df.sort_values(
        "Mean", ascending=False)[:n_elem]

    # Explode the h_mean list to rows
    preprecessed_df = preprecessed_df.explode("h_mean")

    preprecessed_df["preprocessing"] = preprecessed_df["preprocessing"].str.replace(
        "Reweighing", "RW")
    preprecessed_df["preprocessing"] = preprecessed_df["preprocessing"].str.replace(
        "OptimPreproc", "OP")

    # Add the column (1) experiment/(2) bootstrap_type/ (3)preprocessing which will be used as the x_axis
    preprecessed_df["(1) experiment / (2) bootstrap_type / (3)preprocessing"] = \
        "(1) " + preprecessed_df["experiment"].str.upper() + "\n" \
        + "(2) " + preprecessed_df["bootstrap_type"].str.upper() + "\n" \
        + "(3) " + preprecessed_df["preprocessing"].str.upper()

    return preprecessed_df


@typechecked
def select_configurations(df, dataset: str, classifier: str):
    """
    Selects and preprocesses the dataframe of the following configs :
    baseline, RW, LFR, OP, Fairboost-None, Fairboost-Custom, Fairboost-Default.
    :param df: dataframe to be preprocessed
    :param dataset: the dataset name
    :param classifier: the classifier name
    :return: the preprocessed dataframe
    """
    # select the rows by the classifiers and datasets names
    preprecessed_df = df.loc[(df["dataset"] == dataset)
                             & (df["classifier"] == classifier)]

    # select the top n_elems having the highest mean of the h_mean
    preprecessed_df = preprecessed_df.loc[(preprecessed_df["experiment"] == "preprocessing")
                                          | (preprecessed_df["experiment"] == "baseline")
                                          | ((preprecessed_df["experiment"] == "fairboost")
                                             & (preprecessed_df["preprocessing"] == "LFR,OptimPreproc,Reweighing")
                                             )
                                          ]

    preprecessed_df['Mean'] = preprecessed_df["h_mean"].apply(np.median)

    # Explode the h_mean list to rows
    preprecessed_df = preprecessed_df.explode("h_mean")

    preprecessed_df["preprocessing"] = preprecessed_df["preprocessing"].str.replace(
        "Reweighing", "RW")
    preprecessed_df["preprocessing"] = preprecessed_df["preprocessing"].str.replace(
        "OptimPreproc", "OP")

    # Add the column (1) experiment/(2) bootstrap_type/ (3)preprocessing which will be used as the x_axis
    preprecessed_df["(1) experiment / (2) bootstrap_type / (3)preprocessing"] = \
        "(1) " + preprecessed_df["experiment"].str.upper() + "\n" \
        + "(2) " + preprecessed_df["bootstrap_type"].str.upper() + "\n" \
        + "(3) " + preprecessed_df["preprocessing"].str.upper()

    preprecessed_df.loc[preprecessed_df["(1) experiment / (2) bootstrap_type / (3)preprocessing"].str.contains
                        ('BASELINE'), "(1) experiment / (2) bootstrap_type / (3)preprocessing"] = 'BASELINE'

    preprecessed_df.loc[preprecessed_df["(1) experiment / (2) bootstrap_type / (3)preprocessing"].str.contains
                        ('PREPROCESSING & RW'),
                        "(1) experiment / (2) bootstrap_type / (3)preprocessing"] = 'RW'

    preprecessed_df.loc[preprecessed_df["(1) experiment / (2) bootstrap_type / (3)preprocessing"].str.contains
                        ('PREPROCESSING & LFR'),
                        "(1) experiment / (2) bootstrap_type / (3)preprocessing"] = 'LFR'

    preprecessed_df.loc[preprecessed_df["(1) experiment / (2) bootstrap_type / (3)preprocessing"].str.contains
                        ('PREPROCESSING & OP'),
                        "(1) experiment / (2) bootstrap_type / (3)preprocessing"] = 'OP'

    return preprecessed_df


@typechecked
def select_fairboost_configs(df, dataset: str, classifier: str, n_elem=5):
    """
    Selects and preprocesses the dataframe of Fairboost configs.
    :param df: dataframe to be preprocessed
    :param dataset: the dataset name
    :param classifier: the classifier name
    :param n_elem: the number of configuration to analyse (e.g 5 means the top 5 configs)
    :return: the preprocessed dataframe
    """
    # select the rows by the classifiers and datasets names
    preprecessed_df = df.loc[
        (df["experiment"] == "fairboost") & (df["dataset"] == dataset) & (df["classifier"] == classifier)]

    # select the top n_elems having the highest mean of the h_mean
    preprecessed_df['Mean'] = preprecessed_df["h_mean"].apply(np.median)
    preprecessed_df = preprecessed_df.sort_values(
        "Mean", ascending=False)[:n_elem]

    # Explode the h_mean list to rows
    preprecessed_df = preprecessed_df.explode("h_mean")

    preprecessed_df["preprocessing"] = preprecessed_df["preprocessing"].str.replace(
        "Reweighing", "RW")
    preprecessed_df["preprocessing"] = preprecessed_df["preprocessing"].str.replace(
        "OptimPreproc", "OP")

    # Add the column (1) experiment/(2) bootstrap_type/ (3)preprocessing which will be used as the x_axis
    preprecessed_df["(1) bootstrap_type / (2)preprocessing"] = \
        "(1) " + preprecessed_df["bootstrap_type"].str.upper() + "\n" \
        + "(2) " + preprecessed_df["preprocessing"].str.upper()

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
            plot_df = preprocess_dataframe(
                df=df, dataset=dataset, classifier=classifier, n_elem=5)

            plot_title = "h_mean distribution of " + classifier + \
                " model \n trained on the dataset " + dataset

            # sns.set(style="darkgrid")
            fig, ax = plt.subplots(figsize=(16, 13))
            # ax.set(ylim=(.5, 1))

            sns.set_context("paper", font_scale=2, rc={
                            "font.size": 20, "axes.titlesize": 20, "axes.labelsize": 20})
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
            # this function plots the n top elements
            # plot_df = preprocess_dataframe(df=df, dataset=dataset, classifier=classifier, n_elem=5)

            # this function plots the baseline, RW, LFR, OP, Fairboost-None, Fairboost-Custom, Fairboost-Default
            # plot_df = select_configurations(df=df, dataset=dataset, classifier=classifier)

            # this function plots the configurations of Fairboost
            plot_df = select_fairboost_configs(
                df=df, dataset=dataset, classifier=classifier)

            frames.append(plot_df)

    results_df = pd.concat(frames)
    results_df = results_df.drop(
        ['experiment', 'bootstrap_type', 'preprocessing', 'Mean'], axis=1)

    fig, ax = plt.subplots(figsize=(50, 50))

    PROPS = {
        'boxprops': {'facecolor': 'none', 'edgecolor': 'black'},
        'medianprops': {'color': 'black'},
        'whiskerprops': {'color': 'black'},
        'capprops': {'color': 'black'}
    }
    grid = sns.FacetGrid(results_df, col="classifier", row="dataset",
                         height=3, aspect=4, sharey=True, sharex=False)
    grid.map(sns.boxplot, "(1) bootstrap_type / (2)preprocessing", "h_mean", linewidth=1, width=0.5,
             **PROPS)

    # plt.savefig("Boxplots/Combined_plots_classifier_as_rows_scaled_y.pdf")
    plt.savefig("Boxplots/fairboost.pdf")


def main():
    df = get_data_h_mean()
    plot_multiple_boxplots(df)


if __name__ == "__main__":
    main()
