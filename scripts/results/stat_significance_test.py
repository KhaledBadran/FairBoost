import numpy as np
import pandas as pd
import pathlib
import scipy.stats as stats

data_dir = pathlib.Path('statistical_test')
data_file = pathlib.Path(data_dir, 'statistical_tests_raw_data.csv')


def main():

    # Load data
    df = pd.read_csv(data_file)

    # run mannwhitneyu test
    # stats.mannwhitneyu()


if __name__ == '__main__':
    main()