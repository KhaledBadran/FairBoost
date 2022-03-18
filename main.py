from sklearn.tree import DecisionTreeClassifier

from FairBoost import FairBoost
from data import get_german_dataset, get_optim_options
from preprocessing_functions import generate_lambda_function_dir, generate_lambda_function_reweighing, \
    generate_lambda_function_LFR, generate_lambda_function_optimized_preprocessing

dataset_orig_train, dataset_orig_val, dataset_orig_test, unprivileged_groups, privileged_groups = get_german_dataset()
X = dataset_orig_train.features
y = dataset_orig_train.labels.ravel()

model = DecisionTreeClassifier(class_weight='balanced')

preprocessing1 = generate_lambda_function_reweighing(dataset_orig_train, unprivileged_groups, privileged_groups)
preprocessing2 = generate_lambda_function_dir(dataset_orig_train, dataset_orig_train.protected_attribute_names)
preprocessing3 = generate_lambda_function_LFR(dataset_orig_train, unprivileged_groups, privileged_groups)
# optim_options = get_optim_options(dataset_name, len(dataset_orig_train.protected_attribute_names))
# preprocessing4 = generate_lambda_function_optimized_preprocessing(dataset_orig_train, unprivileged_groups,
#                                                                   privileged_groups, optim_options)
preprocessing = (preprocessing1, preprocessing2, preprocessing3)

# data = {'X': X, 'y': y}
ens = FairBoost(model, preprocessing)

## train the ensemble & view estimates for prediction error ##
ens.fit(X, y)

# predict testing data
preprocessing1 = generate_lambda_function_reweighing(dataset_orig_test, unprivileged_groups, privileged_groups)
preprocessing2 = generate_lambda_function_dir(dataset_orig_test, dataset_orig_test.protected_attribute_names)
preprocessing3 = generate_lambda_function_LFR(dataset_orig_test, unprivileged_groups, privileged_groups)
preprocessing = (preprocessing1, preprocessing2, preprocessing3)
ens.preprocessing_functions = preprocessing
predictions = ens.predict(dataset_orig_test.features)
print(predictions)
