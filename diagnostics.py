import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess
import sys
import importlib.metadata

################## Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


################## Function to get model predictions
def model_predictions(dataframe):
    """
    Read the model from prod_deployment_path and predict for the supplied DataFrame.
    Select the same predictor columns used for training; do not include the target.
    Return value is a list containing all predictions.
    """
    with open('config.json', 'r') as f:
        current_config = json.load(f)
    cur_prod_path = current_config['prod_deployment_path']

    if isinstance(dataframe, str):
        dataframe = pd.read_csv(dataframe)

    model_file = os.path.join(cur_prod_path, 'trainedmodel.pkl')
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    feature_cols = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']
    X = dataframe[feature_cols]
    predictions = model.predict(X)
    return predictions.tolist()


################## Function to get summary statistics
def dataframe_summary():
    """
    Read the ingested dataset from output_folder_path. For each numeric column,
    calculate mean, median, and standard deviation.
    Return value is a list containing all summary statistics.
    """
    with open('config.json', 'r') as f:
        current_config = json.load(f)
    cur_dataset_path = current_config['output_folder_path']
    data_file = os.path.join(cur_dataset_path, 'finaldata.csv')
    df = pd.read_csv(data_file)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    summary_statistics = []
    for col in numeric_cols:
        summary_statistics.append(float(df[col].mean()))
        summary_statistics.append(float(df[col].median()))
        summary_statistics.append(float(df[col].std()))

    return summary_statistics


################## Function to check missing data
def missing_data():
    """
    Calculate missing-value percentages for every ingested dataset column,
    including nonnumeric columns, and return a list in column order.
    """
    with open('config.json', 'r') as f:
        current_config = json.load(f)
    cur_dataset_path = current_config['output_folder_path']
    data_file = os.path.join(cur_dataset_path, 'finaldata.csv')
    df = pd.read_csv(data_file)

    na_percentages = [float((df[col].isna().sum() / len(df)) * 100) for col in df.columns]
    return na_percentages


################## Function to get timings
def execution_time():
    """
    Time ingestion.py and training.py, in that order, as required by Step 3
    and the specification. Isolate timing outputs; do not replace the deployed model.
    Return a list of 2 timing values in seconds.
    """
    start_ingestion = timeit.default_timer()
    subprocess.run([sys.executable, 'ingestion.py'], check=True, capture_output=True)
    ingestion_time = timeit.default_timer() - start_ingestion

    start_training = timeit.default_timer()
    subprocess.run([sys.executable, 'training.py'], check=True, capture_output=True)
    training_time = timeit.default_timer() - start_training

    return [float(ingestion_time), float(training_time)]


################## Function to check dependencies
def outdated_packages_list():
    """
    Report every dependency used by the project in a three-column table:
    package name, installed version, and latest available version. Do not upgrade.
    """
    outdated = {}
    try:
        res = subprocess.run([sys.executable, '-m', 'pip', 'list', '--outdated'],
                             capture_output=True, text=True, check=True)
        for line in res.stdout.strip().split('\n')[2:]:
            parts = line.split()
            if len(parts) >= 3:
                outdated[parts[0].lower()] = parts[2]
    except Exception:
        pass

    table = []
    with open('requirements.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            pkg = line.split('==')[0].split('>=')[0].split('<=')[0].strip()
            try:
                installed = importlib.metadata.version(pkg)
            except Exception:
                installed = 'Not Installed'
            latest = outdated.get(pkg.lower(), installed)
            table.append([pkg, str(installed), str(latest)])

    return table


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        cfg = json.load(f)
    test_file = os.path.join(cfg['test_data_path'], 'testdata.csv')
    test_df = pd.read_csv(test_file)

    print("Model predictions:", model_predictions(test_df))
    print("Dataframe summary:", dataframe_summary())
    print("Missing data percentages:", missing_data())
    print("Execution times:", execution_time())
    print("Dependencies table:\n", outdated_packages_list())
