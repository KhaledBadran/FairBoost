import json
import statistics
from typing import List, Dict

import pandas as pd
import numpy as np
from caffe2.perfkernels.hp_emblookup_codegen import opts
from pandas.api.types import CategoricalDtype
from plotnine import *
from plotnine.data import mpg
from typeguard import typechecked


def read_data_baseline(path):
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
    dict = {}
    with open(path, 'r') as f:
        results = json.load(f)
    for dataset_key, dataset_value in results["results"].items():
        for preprocessing_key, preprocessing_value in dataset_value.items():
            for i in range(len(preprocessing_value)):
                for classifier_key, classifier_value in preprocessing_value[i]["results"].items():
                    if preprocessing_key == "fairboost":
                        key = dataset_key + "-" + preprocessing_key + "-" + classifier_key + "-" \
                              + preprocessing_value[i]["hyperparameters"]["init"]['bootstrap_type']
                        dict[key] = classifier_value
                    else:
                        key = dataset_key + "-" + preprocessing_key + "-" + classifier_key + "-" \
                              + preprocessing_value[i]["hyperparameters"]['bootstrap_type']
                        dict[key] = classifier_value
    return dict


@typechecked
def generate_data(data: Dict, dataset_name="", classifier_name=""):
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
            mean_fairness = np.mean(value["disparate_impact"])
            std_accuracy = np.std(value["accuracy"])
            std_fairness = np.std(value["disparate_impact"])
            x1.append(mean_accuracy - (std_accuracy / 2))
            x2.append(mean_accuracy + (std_accuracy / 2))
            y1.append(mean_fairness - (std_fairness / 2))
            y2.append(mean_fairness + (std_fairness / 2))
            # t.append(key.split("-")[1])
            t.append(key)
    d = pd.DataFrame({"x1": x1, "x2": x2, "y1": y1, "y2": y2, "t": t, "r": t})
    return d


@typechecked
def rectangular_plot(data: Dict, plots_path, dataset_name="", classifier_name=""):
    """
    Plots the rectangles plots.
            Parameters:
                    data : List of preprocessing dicts.
            Returns:
                    g (ggplot): returns the plot
    """
    dataframe = generate_data(data, dataset_name=dataset_name, classifier_name=classifier_name)
    plot_title = "Stability of the accuracy and fairness of \n" + \
                 classifier_name + " trained on " + dataset_name + " dataset"

    g = (ggplot(dataframe)
         + scale_x_continuous(name="accuracy")
         + scale_y_continuous(name="fairness")
         # + scale_y_continuous(name="fairness", limits=(0,1.2))
         + geom_rect(data=dataframe, mapping=aes(xmin=dataframe["x1"], xmax=dataframe["x2"], ymin=dataframe["y1"],
                                                 ymax=dataframe["y2"], fill=dataframe["t"]),alpha=1)
                     # color="black", alpha=1)
         # + geom_text(aes(x=dataframe["x1"] + (dataframe["x2"] - dataframe["x1"]) / 2,
         #                 y=dataframe["y1"] + (dataframe["y2"] - dataframe["y1"]) / 2, label=dataframe["r"]),
         #             data=dataframe, size=5)
         + labs(title=plot_title, fill='Preprocessing')
         + theme(legend_margin=-10, legend_box_spacing=0)
         )
    print(g)
    g.save(plots_path+dataset_name+"-"+classifier_name+".png")
    return g


data = read_data_baseline("json_files/baseline_splits.json")
plots_path = "plots/baseline/"

# data = read_data_fairboost("json_files/fairboost_splits.json")
# plots_path = "plots/fairboost/"


datasets = ["german", "adult", "compas"]
classifiers = ["Logistic Regression", "Random Forest"]
for dataset in datasets:
    for classifier in classifiers:
        rectangular_plot(data,plots_path = plots_path, dataset_name=dataset, classifier_name=classifier)
