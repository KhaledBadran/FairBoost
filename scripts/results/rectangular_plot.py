import json
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
from pathlib import Path
import os
import plotnine as pn
from typeguard import typechecked
from math import ceil
import os

import matplotlib.pyplot as plt
from matplotlib import gridspec

from utils import get_plots_dir


def read_data_baseline(path):
    """
    Generates a dict object representing the data in the baseline_splits.json.
            Parameters:
                    path : path of the file baseline_splits.json
            Returns:
                    dict (Dict): a preprocessed dict representing the data
    """
    dict = {}
    with open(path, 'r') as f:
        results = json.load(f)
    for dataset_key, dataset_value in results["results"].items():
        for preprocessing_key, preprocessing_value in dataset_value.items():
            if preprocessing_key == 'baseline':
                for classifier_key, classifier_value in preprocessing_value.items():
                    key = dataset_key + "-" + preprocessing_key + "-" + classifier_key
                    dict[key] = classifier_value
            elif preprocessing_key != "DisparateImpactRemover":
                for classifier_key, classifier_value in preprocessing_value[0]["results"].items():
                    key = dataset_key + "-" + preprocessing_key + "-" + classifier_key
                    dict[key] = classifier_value
    return dict


def read_data_fairboost(path):
    """
    Generates a dict object representing the data in the fairboost_splits.json.
            Parameters:
                    path : path of the file fairboost_splits.json
            Returns:
                    dict (Dict): a preprocessed dict representing the data
    """
    dict = {}
    with open(path, 'r') as f:
        results = json.load(f)
    for dataset_key, dataset_value in results["results"].items():
        for preprocessing_key, preprocessing_value in dataset_value.items():
            if preprocessing_key == "fairboost":
                for i in range(6, len(preprocessing_value), 7):
                    for classifier_key, classifier_value in preprocessing_value[i]["results"].items():
                        key = "Fairboost : " + dataset_key + "-" + preprocessing_key + "-" + classifier_key + "-" \
                              + preprocessing_value[i]["hyperparameters"]["init"]['bootstrap_type']
                        dict[key] = classifier_value
            # else:
            #     for i in range(len(preprocessing_value)):
            #         for classifier_key, classifier_value in preprocessing_value[i]["results"].items():
            #             if preprocessing_value[i]["hyperparameters"]['bootstrap_type'] == "NONE":
            #                 key = "Fairboost : " + dataset_key + "-" + preprocessing_key + "-" + classifier_key + "-" \
            #                       + preprocessing_value[i]["hyperparameters"]['bootstrap_type']
            #                 dict[key] = classifier_value
    return dict


def rename_configs(t):
    """
    Renames the configurations .
            Parameters:
                    t : list of the data to rename
    """
    t = ["Ensembling baseline" if any(j in i for j in [
        "Fairboost : compas-baseline-Random Forest-NONE",
        "Fairboost : compas-baseline-Logistic Regression-NONE",
        "Fairboost : german-baseline-Random Forest-NONE",
        "Fairboost : german-baseline-Logistic Regression-NONE",
        "Fairboost : adult-baseline-Random Forest-NONE",
        "Fairboost : adult-baseline-Logistic Regression-NONE"
    ])
         else i for i in
         t]
    t = ["Fairboost: NONE/LFR,RW,OP" if any(j in i for j in [
        "Fairboost : compas-fairboost-Random Forest-NONE",
        "Fairboost : compas-fairboost-Logistic Regression-NONE"
        "Fairboost : german-fairboost-Random Forest-NONE",
        "Fairboost : german-fairboost-Logistic Regression-NONE",
        "Fairboost : adult-fairboost-Random Forest-NONE",
        "Fairboost : adult-fairboost-Logistic Regression-NONE"
    ]) else i for i in
         t]

    t = ["Fairboost: Default/LFR,RW,OP" if any(j in i for j in [
        "Fairboost : compas-fairboost-Random Forest-DEFAULT",
        "Fairboost : compas-fairboost-Logistic Regression-DEFAULT",
        "Fairboost : german-fairboost-Random Forest-DEFAULT",
        "Fairboost : german-fairboost-Logistic Regression-DEFAULT",
        "Fairboost : adult-fairboost-Random Forest-DEFAULT",
        "Fairboost : adult-fairboost-Logistic Regression-DEFAULT"
    ]) else i for i in
         t]

    t = ["Fairboost: CUSTOM/LFR,RW,OP" if any(j in i for j in [
        "Fairboost : compas-fairboost-Random Forest-CUSTOM",
        "Fairboost : compas-fairboost-Logistic Regression-CUSTOM",
        "Fairboost : german-fairboost-Random Forest-CUSTOM",
        "Fairboost : german-fairboost-Logistic Regression-CUSTOM",
        "Fairboost : adult-fairboost-Random Forest-CUSTOM",
        "Fairboost : adult-fairboost-Logistic Regression-CUSTOM"
    ]) else i for i in
         t]

    t = ["Baseline" if any(j in i for j in ["baseline-Logistic Regression", "baseline-Random Forest"]) else i for i in
         t]
    t = ["Optimized Preprocessing \n(OP)" if any(
        j in i for j in ["OptimPreproc-Logistic Regression", "OptimPreproc-Random Forest"]) else i for i in t]
    t = ["Learning Fair Representation \n (LFR)" if any(
        j in i for j in ["LFR-Logistic Regression", "LFR-Random Forest"]) else i for i in t]
    t = ["Reweighing \n(RW)" if any(
        j in i for j in ["Reweighing-Logistic Regression", "Reweighing-Random Forest"]) else i
         for i in t]
    return t


@typechecked
def to_dataframe(data: Dict, dataset_name="", classifier_name=""):
    """
    Generates a DataFrame object, necessary for the plotting.
            Parameters:
                    data : List of preprocessing dicts.
            Returns:
                    d (DataFrame): returns the dataframe
    """
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    t = []
    for key, value in data.items():
        if (dataset_name in key) and (classifier_name in key):
            mean_accuracy = np.mean(value["accuracy"])
            mean_fairness = np.mean(value["n_disparate_impact"])
            std_accuracy = np.std(value["accuracy"])
            std_fairness = np.std(value["n_disparate_impact"])
            x1.append(mean_accuracy - (std_accuracy / 2))
            x2.append(mean_accuracy + (std_accuracy / 2))
            y1.append(mean_fairness - (std_fairness / 2))
            y2.append(mean_fairness + (std_fairness / 2))
            t.append(key)

    t = rename_configs(t)

    d = pd.DataFrame({"x1": x1, "x2": x2, "y1": y1, "y2": y2, "t": t, "r": t})
    return d


@typechecked
def rectangular_plot(data: Dict, dataset_name="", classifier_name="", print_figures=False,
                     plots_dir=Path("plots/")) -> Tuple:
    """
    Plots the rectangles plots.
            Parameters:
                    data : List of preprocessing dicts.
            Returns:
                    g (ggplot): returns the plot
    """
    dataframe = to_dataframe(
        data, dataset_name=dataset_name, classifier_name=classifier_name)
    plot_title = "Stability of the accuracy and fairness of \n" + \
                 classifier_name + " trained on " + dataset_name + " dataset"

    g = (pn.ggplot(dataframe)
         + pn.scale_x_continuous(name="accuracy")
         + pn.scale_y_continuous(name="fairness")
         # + scale_y_continuous(name="fairness", limits=(0,1.2))
         + pn.geom_rect(data=dataframe, mapping=pn.aes(xmin=dataframe["x1"], xmax=dataframe["x2"], ymin=dataframe["y1"],
                                                       ymax=dataframe["y2"], fill=dataframe["t"]), color="black",
                        alpha=0.6)
         # color="black", alpha=1)
         # + geom_text(aes(x=dataframe["x1"] + (dataframe["x2"] - dataframe["x1"]) / 2,
         #                 y=dataframe["y1"] + (dataframe["y2"] - dataframe["y1"]) / 2, label=dataframe["r"]),
         #             data=dataframe, size=5)
         + pn.labs(title=plot_title, fill='Preprocessing')
         + pn.theme(legend_margin=-10, legend_box_spacing=0)
         )
    if print_figures:
        print(g)
    # Save the plots
    file_name = f'{dataset_name}-{classifier_name}.pdf'
    file_path = plots_dir/file_name
    g.save(file_path)
    return g, plot_title


@typechecked
def read_data() -> Dict:
    """
    Read data from files and return its content in dictionnaries.
            Returns:
                    data: the data contained in both files
    """
    file_dir = Path(__file__).parent.resolve()
    data_path = Path(file_dir, "raw_data").resolve()
    fairboost_results_path = Path(data_path, 'fairboost_splits.json')
    baseline_results_path = Path(data_path, 'baseline_splits.json')
    data_baseline = read_data_baseline(baseline_results_path)
    data_fairboost = read_data_fairboost(fairboost_results_path)
    return {**data_baseline, **data_fairboost}


@typechecked
def plot_all(plots: List, plots_title: List, nb_col=2, plots_dir=Path("plots/"), print_figures=False):
    """
    Plots the rectangles plots in one figure.
            Parameters:
                    plots : List of plots to merge in one figure
                    plots_title: Title of each plot
                    nb_col: The number of columns in the figure
                    plots_dir: Where to save the figure
                    print_figures: Whether to print the figures or not
    """
    nb_row = ceil(len(plots) / 2)

    fig = (pn.ggplot() + pn.geom_blank() + pn.theme_void() + pn.theme(figure_size=(11, 8)) +
           pn.theme(legend_margin=-2, legend_box_spacing=0)).draw()
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.80,
                        top=0.9, wspace=0.4, hspace=0.5)
    gs = gridspec.GridSpec(nb_row, nb_col)

    for i, (plot, plot_title) in enumerate(zip(plots, plots_title)):
        r = i // nb_col
        c = i % nb_col
        p = fig.add_subplot(gs[r, c])
        p.set_xlabel('Accuracy', fontsize=7)
        p.set_ylabel('Fairness', fontsize=7)
        p.set_title(plot_title, fontsize=7)
        _ = plot._draw_using_figure(fig, [p])

    file_name = f'rectangular-all.pdf'
    file_path = Path(plots_dir, file_name)
    fig.savefig(file_path, orientation='portrait')
    if print_figures:
        print(fig)


@typechecked
def add_normalized_di(data: Dict):
    """
    Measures and adds the normalized disparate impact to
    the data dictionnary. For further information on 
    normalized disparate impact, refer to our paper.
            Parameters:
                    data : List of preprocessing dicts.
            Returns:
                    data: Data augmented with normalized disparate impact
    """
    for key in data:
        data[key]['n_disparate_impact'] = [
            score if score <= 1 else (score ** -1) for score in data[key]['disparate_impact']
        ]
    return data


def get_rectangular_plot_dir():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    plots_dir = Path(dir_path, "plots", 'rectangular')
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def main():
    data = read_data()
    data = add_normalized_di(data)
    plots_dir = get_plots_dir("rectangular_plot")

    plots, plots_title = [], []
    datasets = ["german", "adult", "compas"]
    classifiers = ["Logistic Regression", "Random Forest"]
    for dataset in datasets:
        for classifier in classifiers:
            p, p_t = rectangular_plot(data, dataset_name=dataset,
                                      classifier_name=classifier, print_figures=False, plots_dir=plots_dir)
            plots.append(p)
            plots_title.append(p_t)
    plot_all(plots, plots_title, plots_dir=plots_dir)


if __name__ == '__main__':
    main()
