import statistics
from typing import List, Dict

import pandas as pd
import numpy as np
from caffe2.perfkernels.hp_emblookup_codegen import opts
from pandas.api.types import CategoricalDtype
from plotnine import *
from plotnine.data import mpg
from typeguard import typechecked

@typechecked
def generate_data(data):
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
    for elm in data:
        mean_accuracy = np.mean(elm["accuracy"])
        mean_fairness = np.mean(elm["fairness"])
        std_accuracy = np.std(elm["accuracy"])
        std_fairness = np.std(elm["fairness"])
        x1.append(mean_accuracy - (std_accuracy / 2))
        x2.append(mean_accuracy + (std_accuracy / 2))
        y1.append(mean_fairness - (std_fairness / 2))
        y2.append(mean_fairness + (std_fairness / 2))
        t.append(elm['type'])
    d = pd.DataFrame({"x1": x1, "x2": x2, "y1": y1, "y2": y2, "t": t, "r": t})
    return d

@typechecked
def rectangular_plot(data : List[Dict]):
    """
    Plots the rectangles plots.
            Parameters:
                    data : List of preprocessing dicts.
            Returns:
                    g (ggplot): returns the plot
    """
    dataframe = generate_data(data)

    g = (ggplot(dataframe) + scale_x_continuous(name="accuracy") + scale_y_continuous(name="fairness")
         + geom_rect(data=dataframe, mapping=aes(xmin=dataframe["x1"], xmax=dataframe["x2"], ymin=dataframe["y1"],
                                                 ymax=dataframe["y2"], fill=dataframe["t"]),
                     color="black", alpha=0.9)
         + geom_text(aes(x=dataframe["x1"] + (dataframe["x2"] - dataframe["x1"]) / 2,
                         y=dataframe["y1"] + (dataframe["y2"] - dataframe["y1"]) / 2, label=dataframe["r"]),
                     data=dataframe, size=10)
         + labs(fill='Preprocessing')
         + theme(legend_margin=-20,legend_box_spacing=0)
         )
    print(g)
    return g


method_1 = {'type': 'LFR', 'accuracy': [0.1, 0.2, 0.5], 'fairness': [1.3, 0.5, 1.0]}
method_2 = {'type': 'reweighing', 'accuracy': [0.4, 0.5, 0.7], 'fairness': [1.1, 0.9, 1.2]}
method_3 = {'type': 'opt', 'accuracy': [0.2, 0.4, 0.35], 'fairness': [0.3, 0.4, 0.1]}
plot_data = [method_1, method_2, method_3]
rectangular_plot(plot_data)
