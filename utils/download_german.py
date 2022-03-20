import requests
response = requests.get(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")
with open(f'../data/german.data', 'w') as f:
    f.write(response.text)
