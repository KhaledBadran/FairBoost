from typing import Dict, List, Tuple
from utils import read_data
import pandas as pd
import numpy as np
from typeguard import typechecked

@typechecked
def average_performances(results) -> Tuple[np.float64, np.float64]:
    f1_score, di = [], []
    for x in results:
        f1_score = [*f1_score, *x['metrics']['f1-score']]
        di = [*di, *x['metrics']['disparate_impact']]
    normalized_di_score = [
            score if score <= 1 else (score ** -1) for score in di
        ]
    return np.mean(f1_score), np.mean(normalized_di_score)

@typechecked
def get_rows(data:List, column_name)-> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Measures average performance and fairness on the datasets.
            Parameters:
                    data : the data that will be averaged per dataset
                    column_name: the name of the column from the final table
            Returns:
                    (performance, fairness) (Tuple): a tuple of columns wih average performances on datasets
    """
    german = list(filter(lambda x: x['dataset'] == 'german', data))
    compas = list(filter(lambda x: x['dataset'] == 'compas', data))
    adult = list(filter(lambda x: x['dataset'] == 'adult', data))

    performance, fairness = {}, {}
    performance['german'], fairness['german'] = average_performances(german)
    performance['compas'], fairness['compas']  = average_performances(compas)
    performance['adult'], fairness['adult'] = average_performances(adult)
    performance['average'], fairness['average'] = average_performances(data)

    return pd.DataFrame.from_dict(performance,orient='index', columns=[column_name]), pd.DataFrame.from_dict(fairness,orient='index', columns=[column_name])

# The following functions create a column in the final table
@typechecked
def get_baseline_column(data: List) -> Tuple[pd.DataFrame, pd.DataFrame]:
    baseline = list(filter(lambda x : x['experiment'] == "baseline", data))
    return get_rows(baseline, 'Baseline')
@typechecked
def get_LFR_column(data: List) -> Tuple[pd.DataFrame, pd.DataFrame]:
    LFR = list(filter(lambda x : x['experiment'] == "preprocessing" and x['bootstrap_type'] == 'No bootstrap' and 'LFR' in x['preprocessing'], data))
    return get_rows(LFR, 'LFR')
@typechecked
def get_OP_column(data: List) -> Tuple[pd.DataFrame, pd.DataFrame]:
    OP = list(filter(lambda x : x['experiment'] == "preprocessing" and x['bootstrap_type'] == 'No bootstrap' and 'OptimPreproc' in x['preprocessing'], data))
    return get_rows(OP, 'OP')
@typechecked
def get_RW_column(data: List) -> Tuple[pd.DataFrame, pd.DataFrame]:
    RW = list(filter(lambda x : x['experiment'] == "preprocessing" and x['bootstrap_type'] == 'No bootstrap' and 'Reweighing' in x['preprocessing'], data))
    return get_rows(RW, 'RW')
@typechecked
def get_none_1_column(data: List) -> Tuple[pd.DataFrame, pd.DataFrame]:
    x = list(filter(lambda x : x['experiment'] == "fairboost" and x['bootstrap_type'] == 'none' and len(x['preprocessing']) == 1, data))
    return get_rows(x, 'NONE-1')
@typechecked
def get_none_2_column(data: List) -> Tuple[pd.DataFrame, pd.DataFrame]:
    x = list(filter(lambda x : x['experiment'] == "fairboost" and x['bootstrap_type'] == 'none' and len(x['preprocessing']) == 2, data))
    return get_rows(x, 'NONE-2')
@typechecked
def get_none_3_column(data: List) -> Tuple[pd.DataFrame, pd.DataFrame]:
    x = list(filter(lambda x : x['experiment'] == "fairboost" and x['bootstrap_type'] == 'none' and len(x['preprocessing']) == 3, data))
    return get_rows(x, 'NONE-3')
@typechecked
def get_default_1_column(data: List) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = list(filter(lambda x : x['experiment'] == "fairboost" and x['bootstrap_type'] == 'default' and len(x['preprocessing']) == 1, data))
    return get_rows(data, 'DEFAULT-1')
@typechecked
def get_default_2_column(data: List) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = list(filter(lambda x : x['experiment'] == "fairboost" and x['bootstrap_type'] == 'default' and len(x['preprocessing']) == 2, data))
    return get_rows(data, 'DEFAULT-2')
@typechecked
def get_default_3_column(data: List) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = list(filter(lambda x : x['experiment'] == "fairboost" and x['bootstrap_type'] == 'default' and len(x['preprocessing']) == 3, data))
    return get_rows(data, 'DEFAULT-3')
@typechecked
def get_custom_1_column(data: List) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = list(filter(lambda x : x['experiment'] == "fairboost" and x['bootstrap_type'] == 'custom' and len(x['preprocessing']) == 1, data))
    return get_rows(data, 'CUSTOM-1')
@typechecked
def get_custom_2_column(data: List) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = list(filter(lambda x : x['experiment'] == "fairboost" and x['bootstrap_type'] == 'custom' and len(x['preprocessing']) == 2, data))
    return get_rows(data, 'CUSTOM-2')
@typechecked
def get_custom_3_column(data: List) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = list(filter(lambda x : x['experiment'] == "fairboost" and x['bootstrap_type'] == 'custom' and len(x['preprocessing']) == 3, data))
    return get_rows(data, 'CUSTOM-3')
@typechecked
def get_none_0_column(data: List) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = list(filter(lambda x : x['experiment'] == "ensemble" and x['bootstrap_type'] == 'none', data))
    return get_rows(data, 'NONE-0')
@typechecked
def get_default_0_column(data: List) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = list(filter(lambda x : x['experiment'] == "ensemble" and x['bootstrap_type'] == 'default', data))
    return get_rows(data, 'DEFAULT-0')
@typechecked
def get_custom_0_column(data: List) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = list(filter(lambda x : x['experiment'] == "ensemble" and x['bootstrap_type'] == 'custom' , data))
    return get_rows(data, 'CUSTOM-0')


@typechecked
def get_tables(data: List) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns a table of results.
            Parameters:
                    data : all the data from the experiments (baseline_splits.json and fairboost_splits.json)
            Returns:
                    (performance, fairness) (Tuple): a tuple of tables wih average performances on datasets
    """
    funcs = [get_baseline_column, get_LFR_column, get_OP_column, get_RW_column, get_none_0_column, get_none_1_column, get_none_2_column,get_none_3_column,get_default_0_column, get_default_1_column,get_default_2_column,get_default_3_column,get_custom_0_column, get_custom_1_column,get_custom_2_column,get_custom_3_column]
    performance, fairness = [], []
    for func in funcs:
        x = func(data)
        performance.append(x[0])
        fairness.append(x[1])
    return pd.concat(performance, axis=1), pd.concat(fairness, axis=1)



def main():
    data = read_data()
    tables = get_tables(data)
    print(tables[0].round(decimals=3).to_latex())
    print(tables[1].round(decimals=3).to_latex())


if __name__ == "__main__":
    main()