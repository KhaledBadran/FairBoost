import json
from tkinter import font
from typing import List, Dict

import pandas as pd
import numpy as np
from pathlib import Path
import plotnine as pn 

import seaborn as sns
import matplotlib.pyplot as  plt
from matplotlib import gridspec

import os

from typeguard import typechecked
import re

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
            mean_fairness = np.mean(value["disparate_impact"])
            std_accuracy = np.std(value["accuracy"])
            std_fairness = np.std(value["disparate_impact"])
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
    plots_dir.mkdir(parents=True, exist_ok=True)
    file_name = f'rectangular-{dataset_name}-{classifier_name}.pdf'
    file_path = Path(plots_dir, file_name)
    g.save(file_path)
    return g


@typechecked
def rectangular_plot_all(data: Dict, dataset_names=[], classifier_names=[], print_figures=False, plots_dir=Path("plots/")):
    """
    Plots the rectangles plots.
            Parameters:
                    data : List of preprocessing dicts.
            Returns:
                    g (ggplot): returns the plot
    """
    
    plots_  = []
    plots_t = []
    for dataset in dataset_names:
        for classifier in classifier_names:
            dataframe = to_dataframe(data, dataset_name=dataset, classifier_name=classifier)
            dataframe = dataframe.loc[dataframe['t']!='Fairboost : adult-baseline-Logistic Regression-NONE']  
            dataframe = dataframe.loc[dataframe['t']!='Fairboost : german-baseline-Logistic Regression-NONE']  
            dataframe = dataframe.loc[dataframe['t']!='Fairboost : compas-baseline-Logistic Regression-NONE']  
            dataframe = dataframe.loc[dataframe['t']!='Fairboost : german-baseline-Random Forest-NONE']  
            dataframe = dataframe.loc[dataframe['t']!='Fairboost : adult-baseline-Random Forest-NONE']  
            dataframe = dataframe.loc[dataframe['t']!='Fairboost : compas-baseline-Random Forest-NONE']  
            dataframe['t'] = dataframe['t'].str.replace("compas-", " " )
            dataframe['t'] = dataframe['t'].str.replace("adult-", " " )
            dataframe['t'] = dataframe['t'].str.replace("german-", " " )
            dataframe['t'] = dataframe['t'].str.replace("-Random Forest", " " )
            dataframe['t'] = dataframe['t'].str.replace("-Logistic Regression", " " )
            #dataframe['t'] = dataframe['t'].apply(lambda x: re.sub('compas-$','xxxx',x))

            
            plot_title = "Stability of the accuracy and fairness of \n" + \
                     classifier + " trained on " + dataset + " dataset"
            g = (pn.ggplot(dataframe)
                + pn.scale_x_continuous(name="accuracy")
                + pn.scale_y_continuous(name="fairness")
                + pn.geom_rect(data=dataframe, mapping=pn.aes(xmin=dataframe["x1"], xmax=dataframe["x2"], ymin=dataframe["y1"],
                                                       ymax=dataframe["y2"], fill=dataframe["t"]), color="black", alpha=0.3)
                + pn.labs(title=plot_title, fill='Preprocessing')
                + pn.theme(legend_margin=-5, legend_box_spacing=0, text=pn.element_text(size=7))
                )
            plots_.append(g)
            plots_t.append(plot_title)
         
    fig = (pn.ggplot()+pn.geom_blank(data=dataframe)+pn.theme_void()+pn.theme(figure_size=(13,9))+
    pn.theme(legend_margin=-2, legend_box_spacing=0)).draw()
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.80, top=0.9, wspace=0.4, hspace=0.5)
    gs = gridspec.GridSpec(3,2)
    fig_rows = 3
    fig_cols = 2 
    nb_figs = 6
    i = 1
    for r in range(0, fig_rows):
        for c in range(0, fig_cols):
            locals()['ax'+ str(i)] = fig.add_subplot(gs[r,c])
            i+=1
       
    for i in range(0 , nb_figs ):
       locals()['ax'+ str(i+1)].set_xlabel('Accuracy', fontsize=7)
       locals()['ax'+ str(i+1)].set_ylabel('Fairness', fontsize=7)
       locals()['ax'+ str(i+1)].set_title(plots_t[i], fontsize=7)
       _ = plots_[i]._draw_using_figure(fig, [locals()['ax'+ str(i+1)]])
        
    file_name = f'rectangular-all.pdf'
    file_path = Path(plots_dir, file_name)
    fig.savefig(file_path)
   
    
    
   
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


def main():
    data = read_data()

    datasets = ["german", "adult", "compas"]
    classifiers = ["Logistic Regression", "Random Forest"]
    for dataset in datasets:
        for classifier in classifiers:
            rectangular_plot(data, dataset_name=dataset,
                             classifier_name=classifier, print_figures=False)

    rectangular_plot_all(data, dataset_names=datasets,
                             classifier_names=classifiers, print_figures=False)

if __name__ == '__main__':
    main()
