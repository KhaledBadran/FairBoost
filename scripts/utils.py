from collections import defaultdict
from typeguard import typechecked
from pathlib import Path
import os
import json
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric, BinaryLabelDatasetMetric
from typing import Dict
from sklearn.metrics import accuracy_score, f1_score


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
def get_manual_DI(dataset: BinaryLabelDataset) -> float:
    """
    We do not fully trust AIF360 implementation of metrics.
    Thus, we manually computed it by ourselves.
    """
    # What is the value of the protected attribute that is the protected group (0 or 1)
    value_protected = dataset.unprivileged_protected_attributes[0]
    # The value of the protected attribute for each instance
    instance_protected_values = dataset.protected_attributes.ravel()
    y_pred = dataset.labels

    y_protected = y_pred[instance_protected_values == value_protected]
    y_unprotected = y_pred[instance_protected_values != value_protected]

    ratio_unprotected = len(y_unprotected[y_unprotected ==
                                          dataset.favorable_label]) / len(y_unprotected)
    ratio_protected = len(y_protected[y_protected ==
                                      dataset.favorable_label]) / len(y_protected)

    return ratio_protected / ratio_unprotected


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

    classification_metric_bin = BinaryLabelDatasetMetric(
        dataset=classified_dataset,
        unprivileged_groups=dataset_info["unprivileged_groups"],
        privileged_groups=dataset_info["privileged_groups"],
    )

    m_disparate_impact = get_manual_DI(classified_dataset)
    # calculate metrics
    accuracy = accuracy_score(test_dataset.labels, classified_dataset.labels)
    f1 = f1_score(test_dataset.labels, classified_dataset.labels)
    disparate_impact = classification_metric_bin.disparate_impact()
    average_odds_difference = classification_metric.average_odds_difference()

    # print(f"accuracy {accuracy}")
    # print(f"f1-score {f1}")
    # print(f"disparate_impact {disparate_impact}")
    # print(f"Manual disparate impact {m_disparate_impact}")
    # print(f"average odds difference {average_odds_difference}")

    return {
        "accuracy": accuracy,
        "f1-score": f1,
        "disparate_impact": disparate_impact,
        "m_disparate_impact": m_disparate_impact,
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
