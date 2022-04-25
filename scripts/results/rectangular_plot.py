import json
from typing import List, Dict
import pandas as pd
import numpy as np
from pathlib import Path
import os
import plotnine as pn
from typeguard import typechecked


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
            else:
                for i in range(len(preprocessing_value)):
                    for classifier_key, classifier_value in preprocessing_value[i]["results"].items():
                        if preprocessing_value[i]["hyperparameters"]['bootstrap_type'] == "NONE":
                            key = "Fairboost : " + dataset_key + "-" + preprocessing_key + "-" + classifier_key + "-" \
                                  + preprocessing_value[i]["hyperparameters"]['bootstrap_type']
                            dict[key] = classifier_value
    return dict


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
    d = pd.DataFrame({"x1": x1, "x2": x2, "y1": y1, "y2": y2, "t": t, "r": t})
    return d


@typechecked
def rectangular_plot(data: Dict, dataset_name="", classifier_name="", print_figures=False, plots_dir=Path("plots/")):
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
                                                       ymax=dataframe["y2"], fill=dataframe["t"]), color="black", alpha=0.6)
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
    return g


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
            score if score <= 1 else (score**-1) for score in data[key]['disparate_impact']
        ]
    return data

def get_rectangular_plot_dir():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    results_dir = Path(dir_path, "plots", "rectangular_plot")
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def main():
    data = read_data()
    data = add_normalized_di(data)
    plots_dir = get_rectangular_plot_dir()

    datasets = ["german", "adult", "compas"]
    classifiers = ["Logistic Regression", "Random Forest"]
    for dataset in datasets:
        for classifier in classifiers:
            rectangular_plot(data, dataset_name=dataset,
                             classifier_name=classifier, print_figures=False, plots_dir=plots_dir)


if __name__ == '__main__':
    main()
