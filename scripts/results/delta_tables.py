

from typing import List, Set, Tuple
import pandas as pd
from typeguard import typechecked
from utils import read_data
import numpy as np

# -------------------- Fetch functions --------------------


@typechecked
def get_LFR_results(data: List) -> List:
    return list(filter(lambda x: x['experiment'] == "preprocessing" and x['bootstrap_type']
                       == 'No bootstrap' and 'LFR' in x['preprocessing'], data))


@typechecked
def get_OP_results(data: List) -> List:
    return list(filter(lambda x: x['experiment'] == "preprocessing" and x['bootstrap_type']
                       == 'No bootstrap' and 'OptimPreproc' in x['preprocessing'], data))


@typechecked
def get_RW_results(data: List) -> List:
    return list(filter(lambda x: x['experiment'] == "preprocessing" and x['bootstrap_type']
                       == 'No bootstrap' and 'Reweighing' in x['preprocessing'], data))


@typechecked
def get_none_results_by_algo(data: List, algos: Set) -> List:
    return list(filter(lambda x: x['experiment'] == "fairboost" and x['bootstrap_type'] == 'none' and set(x['preprocessing']) == algos, data))


@typechecked
def get_custom_results_by_algo(data: List, algos: Set) -> List:
    return list(filter(lambda x: x['experiment'] == "fairboost" and x['bootstrap_type'] == 'custom' and set(x['preprocessing']) == algos, data))


def aggregate_performances(arrays, metric, aggregator=np.min):
    arrays = np.array([arr['metrics'][metric] for arr in arrays])
    return aggregator(arrays, axis=0)


def get_RW_LFR_results(data: List) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Get data LFR
    lfr_results = get_LFR_results(data)
    # Get data RW
    rw_results = get_RW_results(data)
    # Get data LFR + RW
    ens_results = get_none_results_by_algo(
        data, algos=set(['LFR', 'Reweighing']))
    return get_deltas_all_datasets([*rw_results, *lfr_results], ens_results, column_name='RW_LFR')


def get_RW_OP_results(data: List) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Get data LFR
    op_results = get_OP_results(data)
    # Get data RW
    rw_results = get_RW_results(data)
    # Get data LFR + RW
    ens_results = get_none_results_by_algo(
        data, algos=set(['OptimPreproc', 'Reweighing']))
    return get_deltas_all_datasets([*rw_results, *op_results], ens_results, column_name='OP_RW')


def get_OP_LFR_results(data: List) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Get data LFR
    lfr_results = get_LFR_results(data)
    # Get data RW
    op_results = get_OP_results(data)
    # Get data LFR + RW
    ens_results = get_none_results_by_algo(
        data, algos=set(['LFR', 'OptimPreproc']))
    return get_deltas_all_datasets([*lfr_results, *op_results], ens_results, column_name='OP_LFR')


def get_OP_LFR_RW_results(data: List) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Get data LFR
    lfr_results = get_LFR_results(data)
    # Get data RW
    op_results = get_OP_results(data)
    # Get data RW
    rw_results = get_RW_results(data)

    # Get data LFR + RW
    ens_results = get_none_results_by_algo(
        data, algos=set(['LFR', 'OptimPreproc', 'Reweighing']))
    return get_deltas_all_datasets([*lfr_results, *op_results, *rw_results], ens_results, column_name='OP_LFR_RW')


def get_deltas(unique_res, ensemble_res):
    # Measure the delta in accuracy, compared to the worst performing method, for each split
    min_accuracy = aggregate_performances(
        unique_res, metric="f1-score", aggregator=np.min)
    accuracy_delta = np.array(
        ensemble_res['metrics']['f1-score']) - min_accuracy

    # Measure the delta in fairness, compared to the worst performing method, for each split
    min_accuracy = aggregate_performances(
        unique_res, metric="disparate_impact", aggregator=np.min)
    fairness_delta = np.array(
        ensemble_res['metrics']['disparate_impact']) - min_accuracy

    accuracy_delta, fairness_delta = np.mean(
        accuracy_delta), np.mean(fairness_delta)
    return accuracy_delta, fairness_delta


@typechecked
def get_deltas_all_datasets(unique_res, ensemble_res: List, column_name) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def filter_dataset(data, dataset_name):
        return list(filter(lambda x: x['dataset'] == dataset_name, data))

    performance, fairness = {}, {}
    performance['german'], fairness['german'] = get_deltas(filter_dataset(
        unique_res, 'german'), filter_dataset(ensemble_res, 'german')[0])
    performance['compas'], fairness['compas'] = get_deltas(filter_dataset(
        unique_res, 'compas'), filter_dataset(ensemble_res, 'compas')[0])
    performance['adult'], fairness['adult'] = get_deltas(filter_dataset(
        unique_res, 'adult'), filter_dataset(ensemble_res, 'adult')[0])
    performance['average'], fairness['average'] = np.mean([performance['german'], performance['compas'], performance['adult']]), np.mean([
        fairness['german'], fairness['compas'], fairness['adult']])

    return pd.DataFrame.from_dict(performance, orient='index', columns=[column_name]), pd.DataFrame.from_dict(fairness, orient='index', columns=[column_name])


def get_table(data, classifier):
    data = list(filter(lambda x: x['classifier'] == classifier, data))
    funcs = [get_RW_LFR_results, get_RW_OP_results,
             get_OP_LFR_results, get_OP_LFR_RW_results]
    performance, fairness = [], []
    for func in funcs:
        x = func(data)
        performance.append(x[0])
        fairness.append(x[1])
    return pd.concat(performance, axis=1), pd.concat(fairness, axis=1)


if __name__ == "__main__":
    data = read_data()
    table_performance, table_fairness = get_table(
        data, classifier='Random Forest')
    print(table_performance.round(decimals=3).to_latex())
    print(table_fairness.round(decimals=3).to_latex())
