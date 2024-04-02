from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

################# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])


################# Function for model scoring
def score_model(model=None, dataframe=None, score_path=None):
    """
    Calculate and return the supplied model's F1 score on the supplied DataFrame.
    Step 2: use the output_model_path model and test_data_path/testdata.csv.
    API: use the deployed model and test_data_path/testdata.csv.
    Drift check: use the same deployed model and newly ingested data.
    Write latestscore.txt only when score_path is supplied; API and drift checks
    must not overwrite the deployed baseline score.
    """
    with open('config.json', 'r') as f:
        current_config = json.load(f)

    # Load model if not provided
    if model is None:
        cur_model_dir = current_config['output_model_path']
        model_file = os.path.join(cur_model_dir, 'trainedmodel.pkl')
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
    elif isinstance(model, str):
        with open(model, 'rb') as f:
            model = pickle.load(f)

    # Load dataframe if not provided
    if dataframe is None:
        cur_test_dir = current_config['test_data_path']
        test_file = os.path.join(cur_test_dir, 'testdata.csv')
        dataframe = pd.read_csv(test_file)
    elif isinstance(dataframe, str):
        dataframe = pd.read_csv(dataframe)

    feature_cols = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']
    X = dataframe[feature_cols]
    y = dataframe['exited']

    predictions = model.predict(X)
    f1 = float(metrics.f1_score(y, predictions))

    if score_path is not None:
        os.makedirs(os.path.dirname(score_path), exist_ok=True)
        with open(score_path, 'w') as f:
            f.write(str(f1))

    return f1


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        cfg = json.load(f)
    out_model_dir = cfg['output_model_path']
    test_dir = cfg['test_data_path']

    with open(os.path.join(out_model_dir, 'trainedmodel.pkl'), 'rb') as f:
        trained_model = pickle.load(f)
    test_df = pd.read_csv(os.path.join(test_dir, 'testdata.csv'))
    target_score_path = os.path.join(out_model_dir, 'latestscore.txt')

    f1 = score_model(trained_model, test_df, score_path=target_score_path)
    print(f"Scored model F1: {f1}")
