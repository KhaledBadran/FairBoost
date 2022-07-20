from utils import read_data
import pandas as pd
import numpy as np

def average_performances(results):
    f1_score, m_di = [], []
    for x in results:
        f1_score = [*f1_score, *x['metrics']['f1-score']]
        m_di = [*m_di, *x['metrics']['m_disparate_impact']]
    return np.mean(f1_score), np.mean(m_di)

def get_rows(data, column_name):
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
def get_baseline_column(data):
    baseline = list(filter(lambda x : x['experiment'] == "baseline", data))
    return get_rows(baseline, 'baseline')
def get_LFR_column(data):
    LFR = list(filter(lambda x : x['experiment'] == "preprocessing" and x['bootstrap_type'] == 'No bootstrap' and 'LFR' in x['preprocessing'], data))
    return get_rows(LFR, 'LFR')
def get_OP_column(data):
    OP = list(filter(lambda x : x['experiment'] == "preprocessing" and x['bootstrap_type'] == 'No bootstrap' and 'OptimPreproc' in x['preprocessing'], data))
    return get_rows(OP, 'OP')
def get_RW_column(data):
    RW = list(filter(lambda x : x['experiment'] == "preprocessing" and x['bootstrap_type'] == 'No bootstrap' and 'Reweighing' in x['preprocessing'], data))
    return get_rows(RW, 'RW')
def get_none_1_column(data):
    x = list(filter(lambda x : x['experiment'] == "fairboost" and x['bootstrap_type'] == 'none' and len(x['preprocessing']) == 1, data))
    return get_rows(x, 'NONE-1')
def get_none_2_column(data):
    x = list(filter(lambda x : x['experiment'] == "fairboost" and x['bootstrap_type'] == 'none' and len(x['preprocessing']) == 2, data))
    return get_rows(x, 'NONE-2')
def get_none_3_column(data):
    x = list(filter(lambda x : x['experiment'] == "fairboost" and x['bootstrap_type'] == 'none' and len(x['preprocessing']) == 3, data))
    return get_rows(x, 'NONE-3')
def get_default_1_column(data):
    data = list(filter(lambda x : x['experiment'] == "fairboost" and x['bootstrap_type'] == 'default' and len(x['preprocessing']) == 1, data))
    return get_rows(data, 'DEFAULT-1')
def get_default_2_column(data):
    data = list(filter(lambda x : x['experiment'] == "fairboost" and x['bootstrap_type'] == 'default' and len(x['preprocessing']) == 2, data))
    return get_rows(data, 'DEFAULT-2')
def get_default_3_column(data):
    data = list(filter(lambda x : x['experiment'] == "fairboost" and x['bootstrap_type'] == 'default' and len(x['preprocessing']) == 3, data))
    return get_rows(data, 'DEFAULT-3')
def get_custom_1_column(data):
    data = list(filter(lambda x : x['experiment'] == "fairboost" and x['bootstrap_type'] == 'custom' and len(x['preprocessing']) == 1, data))
    return get_rows(data, 'CUSTOM-1')
def get_custom_2_column(data):
    data = list(filter(lambda x : x['experiment'] == "fairboost" and x['bootstrap_type'] == 'custom' and len(x['preprocessing']) == 2, data))
    return get_rows(data, 'CUSTOM-2')
def get_custom_3_column(data):
    data = list(filter(lambda x : x['experiment'] == "fairboost" and x['bootstrap_type'] == 'custom' and len(x['preprocessing']) == 3, data))
    return get_rows(data, 'CUSTOM-3')
def get_baseline_ensemble(data):
    data = list(filter(lambda x : x['experiment'] == "ensemble" , data))
    return get_rows(data, 'ensemble')


def get_tables(data):
    """
    Returns a table of results.
            Parameters:
                    data : all the data from the experiments (baseline_splits.json and fairboost_splits.json)
            Returns:
                    (performance, fairness) (Tuple): a tuple of tables wih average performances on datasets
    """
    funcs = [get_baseline_column, get_LFR_column, get_OP_column, get_RW_column, get_baseline_ensemble, get_none_1_column, get_none_2_column,get_none_3_column,get_default_1_column,get_default_2_column,get_default_3_column,get_custom_1_column,get_custom_2_column,get_custom_3_column]
    performance, fairness = [], []
    for func in funcs:
        x = func(data)
        performance.append(x[0])
        fairness.append(x[1])
    return pd.concat(performance, axis=1), pd.concat(fairness, axis=1)



def main():
    data = read_data()
    tables = get_tables(data)
    print(tables[0].to_latex())
    print(tables[1].to_latex())


if __name__ == "__main__":
    main()