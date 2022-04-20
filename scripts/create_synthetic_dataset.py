import sys

from configs.constants import DATASETS
sys.path.insert(0, '../')

import numpy as np
import pandas as pd


# Datasets
from aif360.datasets import AdultDataset
from aif360.datasets import CompasDataset
from aif360.datasets import GermanDataset

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric, BinaryLabelDatasetMetric

from pathlib import Path
# Iterate each dataset
for dataset_name, dataset_info in DATASETS.items():


     ds = dataset_info["original_dataset"].copy()
     f = open( dataset_name+'.txt', 'w') 
     nb_features = len(ds.features)
     it = 0
     results = []
     print(f'\n Number of rows for {dataset_name} is {nb_features}')
     
     # j[1] index where feacture (sex) is located in the dataset 
     loc_ = 1
     
     if dataset_name =='german':
          print('german')
     elif dataset_name == 'compas':
          print('compas')
          loc_ = 0
     elif dataset_name == 'adult':
          print('adult')


     
     ind_0 = [ i for i,j in enumerate(ds.features) if j[loc_] == ds.unprivileged_protected_attributes[0] ] # female
     #dsx = ds.subset(ind_0)
     #ind_0 = [ i for i,j in enumerate(dsx.labels)  if j[0] == loc_ ] # female
     ind_1 = [ i for i,j in enumerate(ds.features) if j[loc_] == ds.privileged_protected_attributes[0] ] # male
     
     len_index_0 = len(ind_0) # number of indices for female
     
     range_ = len(ind_0) if (nb_features-len_index_0) > len_index_0 else (nb_features-len_index_0 ) # pick equally or less number of male
     for i in  range(range_, 0, -50):
          index_ =  (list(set().union( ind_0, ind_1[0: len_index_0]))) # indexes to subset the dataset
          subset_ds = ds.subset(index_) 
          
          classification_metric_bin = BinaryLabelDatasetMetric(dataset=subset_ds,
          unprivileged_groups = dataset_info["unprivileged_groups"],
          privileged_groups = dataset_info["privileged_groups"],
          )

               
          di = classification_metric_bin.disparate_impact()
          
          if di > 1.33:
              _biased ='bias towards unprivileged'
          elif di < 0.75:
              _biased ='bias towards privileged'
          elif di > 0.95 and di < 1.052:
              _biased ='fair'
          else:
              _biased ='disparate impact out of the established ratio'
          
          f.write(str(i)+','+  str(di) + ',' + _biased +'\n')
          
   