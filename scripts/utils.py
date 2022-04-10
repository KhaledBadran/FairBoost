from collections import defaultdict
from typeguard import typechecked
from pathlib import Path
import os
import json
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from typing import Dict
from sklearn.metrics import accuracy_score


@typechecked
def save_results(filename: str, results: Dict, experiment_details: Dict = {}):
    """
    Saves a given dictionary to a JSON file.

    :param experiment_details: configurations in the constants files used to run the experiment (e.g., seeds)
     and date of experiment
    :param filename: The name of the output file
    :param results: The results to be saved
    """

    experiment_results = {
        "experiment_details": experiment_details, "results": results}
    dir_path = os.path.dirname(os.path.realpath(__file__))
    results_dir = Path(dir_path, "results", 'raw_data')
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(f"{results_dir}/{filename}.json", "w") as fp:
        json.dump(experiment_results, fp, indent=4, cls=FunctionEncoder)


@typechecked
def measure_results(
    test_dataset: BinaryLabelDataset,
    classified_dataset: BinaryLabelDataset,
    dataset_info: Dict,
) -> Dict:
    """
    Computes fairness and accuracy metrics.

    :param test_dataset: an AIF360 dataset containing the test examples with their labels
    :param classified_dataset: an AIF360 dataset containing the test examples with the predicted labels
    :param dataset_info: information about the dataset including privileged and unprivileged groups
    :return: a dictionary of accuracy and fairness metrics
    """
    classification_metric = ClassificationMetric(
        dataset=test_dataset,
        classified_dataset=classified_dataset,
        unprivileged_groups=dataset_info["unprivileged_groups"],
        privileged_groups=dataset_info["privileged_groups"],
    )

    # calculate metrics
    accuracy = accuracy_score(test_dataset.labels, classified_dataset.labels)
    disparate_impact = classification_metric.disparate_impact()
    average_odds_difference = classification_metric.average_odds_difference()

    print(f"accuracy {accuracy}")
    print(f"disparate_impact {disparate_impact}")
    print(f"average odds difference {average_odds_difference}")

    return {
        "accuracy": accuracy,
        "disparate_impact": disparate_impact,
        "average_odds_difference": average_odds_difference,
    }


def merge_results_array(results):
    metrics_agg = defaultdict(lambda: defaultdict(list))
    for result in results:
        for algo_type in result:
            for metric_name in result[algo_type]:
                metrics_agg[algo_type][metric_name].append(
                    result[algo_type][metric_name]
                )
    return metrics_agg


class FunctionEncoder(json.JSONEncoder):
    def default(self, obj):
        if callable(obj):
            return obj.__name__
        return json.JSONEncoder.default(self, obj)
