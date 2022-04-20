import numpy as np
from collections import defaultdict
from datetime import datetime
from aif360.datasets import BinaryLabelDataset

# use constants from constants_best_LFR_params_search file
from configs.constants_best_LFR_params_search import (
    DATASETS,
    CLASSIFIERS,
    SEEDS,
    CLASSIFIERS_HYPERPARAMETERS,
)

from utils import save_results
from evaluation_baseline import evaluate_mitigation_techniques

# typechecking
from typeguard import typechecked
from typing import Dict, Tuple

np.random.seed(0)

def main():
    results = defaultdict(dict)

    for dataset_name, dataset_info in DATASETS.items():

        print(f"\n\n$$$$in dataset {dataset_name}$$$$$\n")

        dataset: BinaryLabelDataset = dataset_info["original_dataset"]

        print(f"\n\n---------- Unfairness Mitigation techniques ----------")
        results = evaluate_mitigation_techniques(
            results, dataset, dataset_name, dataset_info, seeds=SEEDS)

    # save the results to file
    experiment_details = {
        "DATE": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "CLASSIFIERS_HYPERPARAMETERS": CLASSIFIERS_HYPERPARAMETERS,
        "SEEDS": SEEDS,
    }

    save_results(
        filename="LFR_evaluation",
        results=results,
        experiment_details=experiment_details,
    )


if __name__ == "__main__":
    main()

