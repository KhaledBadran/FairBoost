import pandas as pd
import numpy as np
from typeguard import typechecked
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path

from utils import get_data_h_mean, get_plots_dir


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

    # select specific configs
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

    # Add the column Preprocessing type which will be used as the x_axis
    preprecessed_df["Preprocessing type"] = \
        preprecessed_df["experiment"].str.upper() + " : \n" \
        + preprecessed_df["bootstrap_type"].str.upper() + "\n" \
        + preprecessed_df["preprocessing"].str.upper()

    # Change the names of some elements preprocessing types
    "PREPROCESSING" + "\n" \
        + " : " + preprecessed_df["bootstrap_type"].str.upper() + "\n" \
        + "RW"

    RW = list("PREPROCESSING" + " : \n" +
              preprecessed_df["bootstrap_type"].str.upper() + "\n" + "RW")[0]

    LFR = list("PREPROCESSING" + " : \n" +
               preprecessed_df["bootstrap_type"].str.upper() + "\n" + "LFR")[0]

    OP = list("PREPROCESSING" + " : \n" +
              preprecessed_df["bootstrap_type"].str.upper() + "\n" + "OP")[0]

    preprecessed_df.loc[preprecessed_df["Preprocessing type"].str.contains(
        'BASELINE'), "Preprocessing type"] = 'BASELINE'

    preprecessed_df["Preprocessing type"] = \
        np.where(preprecessed_df["Preprocessing type"] == RW,
                 "Reweighing \n(RW)", preprecessed_df["Preprocessing type"])

    preprecessed_df["Preprocessing type"] = \
        np.where(preprecessed_df["Preprocessing type"] == LFR,
                 "Learning Fair \n Representations (LFR)", preprecessed_df["Preprocessing type"])

    preprecessed_df["Preprocessing type"] = \
        np.where(preprecessed_df["Preprocessing type"] == OP,
                 "Optimized \n Preprocessing \n (OP)", preprecessed_df["Preprocessing type"])

    # Add the column Dataset - Classifier
    preprecessed_df["Dataset - Classifier"] = \
        preprecessed_df["dataset"].str.upper() \
        + " - " + preprecessed_df["classifier"].str.upper()

    preprecessed_df = preprecessed_df.drop(['dataset', 'classifier'], axis=1)
    return preprecessed_df


@typechecked
def select_fairboost_configs(df, dataset: str, classifier: str, n_elem):
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


def plot_unique_boxplot(df, plots_dir=Path("plots/")):
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

            # sns.set_context("paper", font_scale=2, rc={"font.size": 20, "axes.titlesize": 20, "axes.labelsize": 20})
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
            plt.savefig(plots_dir/f"{classifier}_{dataset}.pdf")


def plot_multiple_boxplots(df, scale_y=True, plots_dir=Path("plots/")):
    """
    this method will generate a plot combining all the combinations of classifiers and datasets.
    The plots generated will have the name Combined_plots.pdf
    :param df: the dataframe of the data
    :param scale_y: a boolean indicating if we should scale y axis or not
    """
    classifiers_list = df["classifier"].unique()
    datasets_list = df["dataset"].unique()
    frames = []
    for classifier in classifiers_list:
        for dataset in datasets_list:
            # this function plots the n top elements
            # plot_df = preprocess_dataframe(df=df, dataset=dataset, classifier=classifier, n_elem=5)

            # this function plots the baseline, RW, LFR, OP, Fairboost-None, Fairboost-Custom, Fairboost-Default
            plot_df = select_configurations(
                df=df, dataset=dataset, classifier=classifier)

            frames.append(plot_df)

    results_df = pd.concat(frames)

    # The following lines sorts the data by the name of the classifier
    data_classifier = ["GERMAN - LOGISTIC REGRESSION", "GERMAN - RANDOM FOREST",
                       "ADULT - LOGISTIC REGRESSION", "ADULT - RANDOM FOREST",
                       "COMPAS - LOGISTIC REGRESSION", "COMPAS - RANDOM FOREST"]

    preprocessing_type = ["BASELINE", "Reweighing \n(RW)", "Learning Fair \n Representations (LFR)",
                          "Optimized \n Preprocessing \n (OP)", "FAIRBOOST : \nNONE\nLFR,OP,RW",
                          "FAIRBOOST : \nDEFAULT\nLFR,OP,RW", "FAIRBOOST : \nCUSTOM\nLFR,OP,RW"]

    results_df["Dataset - Classifier"] = results_df["Dataset - Classifier"].astype(
        "category")
    results_df["Dataset - Classifier"].cat.set_categories(
        data_classifier, inplace=True)

    results_df["Preprocessing type"] = results_df["Preprocessing type"].astype(
        "category")
    results_df["Preprocessing type"].cat.set_categories(
        preprocessing_type, inplace=True)

    results_df = results_df.sort_values(
        ["Dataset - Classifier", "Preprocessing type"])

    # We drop the useless columns
    results_df = results_df.drop(
        ['experiment', 'bootstrap_type', 'preprocessing', 'Mean'], axis=1)

    # We plot the boxplot
    fig, ax = plt.subplots(figsize=(50, 40))

    PROPS = {
        'boxprops': {'facecolor': 'none', 'edgecolor': 'black'},
        'medianprops': {'color': 'black'},
        'whiskerprops': {'color': 'black'},
        'capprops': {'color': 'black'}
    }
    grid = sns.FacetGrid(results_df, row="Dataset - Classifier",
                         height=3, aspect=4, sharey=scale_y, sharex=True)
    grid.map(sns.boxplot, "Preprocessing type", "h_mean", linewidth=1, width=0.5,
             **PROPS)

    plt.savefig(plots_dir/"Combined_plots.pdf")


def plot_merged_boxplot(df, plots_dir=Path("plots/")):
    """
    this method will generate a single plot combining all the combinations of classifiers and datasets by merging the
    h_mean of the data
    :param df: the dataframe of the data
    """
    classifiers_list = df["classifier"].unique()
    datasets_list = df["dataset"].unique()
    frames = []
    for classifier in classifiers_list:
        for dataset in datasets_list:
            plot_df = select_configurations(
                df=df, dataset=dataset, classifier=classifier)

            frames.append(plot_df)

    results_df = pd.concat(frames)

    fig, ax = plt.subplots(figsize=(25, 12))
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
    plot = sns.boxplot(x="Preprocessing type", y="h_mean", data=results_df,
                       ax=ax, showfliers=False, linewidth=3, width=0.5, **PROPS)
    # plot.set_xlabel(fontsize=30)
    plt.savefig(plots_dir/"Merged_plots.pdf")

    # preprocessing_type = ["BASELINE", "Reweighing \n(RW)", "Learning Fair \n Representations (LFR)",
    #                       "Optimized \n Preprocessing \n (OP)", "FAIRBOOST : \nNONE\nLFR,OP,RW",
    #                       "FAIRBOOST : \nDEFAULT\nLFR,OP,RW", "FAIRBOOST : \nCUSTOM\nLFR,OP,RW"]
    #
    # results_df["Preprocessing type"] = results_df["Preprocessing type"].astype("category")
    # results_df["Preprocessing type"].cat.set_categories(preprocessing_type, inplace=True)
    #
    # results_df = results_df.sort_values(["Preprocessing type"])
    #
    # # We drop the useless columns
    # results_df = results_df.drop(['experiment', 'bootstrap_type', 'preprocessing', 'Mean', "Dataset - Classifier"],
    #                              axis=1)
    #
    # # We plot the boxplot
    # fig, ax = plt.subplots(figsize=(16, 13))
    # sns.set_context("paper", font_scale=2, rc={"font.size": 20, "axes.titlesize": 20, "axes.labelsize": 20})
    # # sns.set(font_scale=1)
    # PROPS = {
    #     'boxprops': {'facecolor': 'none', 'edgecolor': 'black'},
    #     'medianprops': {'color': 'black'},
    #     'whiskerprops': {'color': 'black'},
    #     'capprops': {'color': 'black'}
    # }
    #
    # plot = sns.boxplot(x="Preprocessing type", y="h_mean", data=results_df, showfliers=False, ax=ax, linewidth=3,
    #                    width=0.5, **PROPS)
    #
    # plt.savefig("Boxplots/Merged_plots.pdf")


def plot_fairboost_boxplots(df, plots_dir=Path("plots/")):
    """
    this method will generate a plot combining all the combinations of classifiers and datasets for fairboost configs.
    The plots generated will have the name Combined_plots.pdf
    :param df: the dataframe of the data
    """
    classifiers_list = df["classifier"].unique()
    datasets_list = df["dataset"].unique()
    frames = []
    for classifier in classifiers_list:
        for dataset in datasets_list:
            plot_df = select_fairboost_configs(
                df=df, dataset=dataset, classifier=classifier, n_elem=7)
            frames.append(plot_df)

    results_df = pd.concat(frames)

    # We drop the useless columns
    results_df = results_df.drop(
        ['experiment', 'bootstrap_type', 'preprocessing', 'Mean'], axis=1)

    # We plot the boxplot
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

    plt.savefig(plots_dir/"fairboost.pdf")


def plot_unique_fairboost_boxplots(df, plots_dir=Path("plots/")):
    """
    this method will generate a plot for every combination of classifiers and datasets for fairboost configurations.
    The plots generated will have the name fairboost_{classifier}_{dataset}.pdf
    :param df: the dataframe of the data
    """
    classifiers_list = df["classifier"].unique()
    datasets_list = df["dataset"].unique()
    frames = []
    for classifier in classifiers_list:
        for dataset in datasets_list:
            plot_df = select_fairboost_configs(
                df=df, dataset=dataset, classifier=classifier, n_elem=8)
            frames.append(plot_df)

            # fig, ax = plt.subplots(figsize=(16, 13))
            fig, ax = plt.subplots(figsize=(25, 12))
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
            plot = sns.boxplot(x="(1) bootstrap_type / (2)preprocessing", y="h_mean", data=plot_df,
                               ax=ax, showfliers=False, linewidth=3, width=0.5, **PROPS)
            # plot.set_xlabel(fontsize=30)
            plt.savefig(plots_dir/f"fairboost_{classifier}_{dataset}.pdf")


def main():
    df = get_data_h_mean() 
    plots_dir = get_plots_dir("boxplots")
    
    # Creates a  unique file for every configuration
    plot_unique_boxplot(df, plots_dir=plots_dir)

    # Creates a file combining all the plots
    plot_multiple_boxplots(df, scale_y=False, plots_dir=plots_dir)

    # Creates the fairboost file
    plot_fairboost_boxplots(df, plots_dir=plots_dir)

    #  Creates a  unique file for every fairboost configuration
    plot_unique_fairboost_boxplots(df, plots_dir=plots_dir)

    # Create the combined boxplot
    plot_merged_boxplot(df, plots_dir=plots_dir)


if __name__ == "__main__":
    main()
