import json
import os
import pandas as pd
from pathlib import Path
from typing import List
from typeguard import typechecked


@typechecked
def contraharmonic_mean(values: List) -> float:
    """
    calcuates the contraharmonic mean of a list of values

    :param values: list of values to calculate the contraharmonic mean of
    :return: contraharmonic mean
    """
    numerator = sum([val ** 2 for val in values])
    denominator = sum(values)
    return numerator / denominator


def main():
    results_dir = 'results'
    file_to_read = Path(results_dir, 'LFR_evaluation.json')
    output_file = Path(results_dir, 'LFR_evaluation.csv')

    with open(file_to_read, "r") as read_file:
        data = json.load(read_file)

    results_to_save = []

    for dataset in data.keys():
        all_params_results = data[dataset]['LFR']

        for param_results in all_params_results:
            hyperparameters = param_results['hyperparameters']
            metrics = param_results['results']

            # metric results can be empty if all labels were same
            if not metrics['Logistic Regression']:
                # print('No results')
                continue
            else:

                LR_results = metrics['Logistic Regression']
                RF_results = metrics['Random Forest']

                results_to_save.append((dataset,
                                        str(hyperparameters),
                                        1 - LR_results['f_score'],
                                        abs(1 - LR_results['disparate_impact']),
                                        1 - RF_results['f_score'],
                                        abs(1 - RF_results['disparate_impact'])),
                                       )

    df = pd.DataFrame(results_to_save,
                      columns=['dataset', 'hyperparameters', 'LR_f1_err', 'LR_DI_err', 'RF_f1_err', 'RF_DI_err'])

    # calculate the average of the metric columns
    df['average_score'] = df[['LR_f1_err', 'LR_DI_err', 'RF_f1_err', 'RF_DI_err']].mean(axis=1)
    df['contraharmonic_mean'] = df.apply(lambda row: contraharmonic_mean([row['LR_f1_err'], row['LR_DI_err'], row['RF_f1_err'], row['RF_DI_err']]), axis = 1)

    df.to_csv(output_file, index=False)

    for dataset in data.keys():
        print(dataset)

        # get the rows of each dataset
        dataset_df = df.loc[df['dataset'] == dataset]

        # find the minimum contraharmonic_mean
        best_param_results = \
            dataset_df[dataset_df['contraharmonic_mean'] == dataset_df['contraharmonic_mean'].min()].to_dict(orient='records')[0]

        # record the hyperparameters and average_score
        print(best_param_results['hyperparameters'], best_param_results['contraharmonic_mean'])


if __name__ == "__main__":
    main()

