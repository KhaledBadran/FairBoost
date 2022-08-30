from utils import read_data
import numpy as np
from collections import defaultdict


def get_results(data,  classifier, dataset, ensemble, bootstrap_type):
    t = list(filter(
        lambda x: x['classifier'] == classifier and x['dataset'] == dataset and set(x['preprocessing']) == ensemble and x['bootstrap_type'] == bootstrap_type, data))
    return t[0]


if __name__ == "__main__":
    # Declaring this function in the main scope so it has access to the variable in this scope (ans)
    def compare_approaches(none_results, default_results, metric_name):
        n_acc = np.mean(none_results['metrics'][metric_name])
        d_acc = np.mean(default_results['metrics'][metric_name])
        if n_acc > d_acc:
            ans[f'none-{metric_name}'] += 1
        else:
            ans[f'default-{metric_name}'] += 1

    data = read_data()

    ans = defaultdict(int)
    for classifier in ['Random Forest', 'Logistic Regression']:
        for dataset in ['german', 'compas', 'adult']:
            for ensemble in [set(['LFR', 'OptimPreproc']), set(['LFR', 'Reweighing']), set(['OptimPreproc', 'Reweighing']), set(['LFR', 'OptimPreproc', 'Reweighing'])]:
                none_results = get_results(
                    data, classifier, dataset, ensemble, 'none')
                default_results = get_results(
                    data, classifier, dataset, ensemble, 'default')

                compare_approaches(
                    none_results, default_results, 'disparate_impact')
                compare_approaches(none_results, default_results, 'f1-score')

    print(ans)
