from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
import shutil
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

################## Load config.json and correct path variable
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


#################### Function for deployment
def store_model_into_pickle(model=None):
    """
    Copy the latest pickle file, the latestscore.txt value,
    and the ingestedfiles.txt file into the deployment directory.
    The score must describe the model being copied, not an earlier model.
    """
    with open('config.json', 'r') as f:
        current_config = json.load(f)

    cur_output_dir = current_config['output_folder_path']
    cur_model_dir = current_config['output_model_path']
    cur_prod_dir = current_config['prod_deployment_path']

    os.makedirs(cur_prod_dir, exist_ok=True)

    # If a model instance is directly passed, write it or copy from cur_model_dir
    src_model_file = os.path.join(cur_model_dir, 'trainedmodel.pkl')
    dst_model_file = os.path.join(cur_prod_dir, 'trainedmodel.pkl')
    if model is not None and not isinstance(model, str):
        with open(dst_model_file, 'wb') as f:
            pickle.dump(model, f)
    elif os.path.exists(src_model_file):
        shutil.copy2(src_model_file, dst_model_file)

    # Copy latestscore.txt
    src_score_file = os.path.join(cur_model_dir, 'latestscore.txt')
    dst_score_file = os.path.join(cur_prod_dir, 'latestscore.txt')
    if os.path.exists(src_score_file):
        shutil.copy2(src_score_file, dst_score_file)

    # Copy ingestedfiles.txt
    src_ingested_file = os.path.join(cur_output_dir, 'ingestedfiles.txt')
    dst_ingested_file = os.path.join(cur_prod_dir, 'ingestedfiles.txt')
    if os.path.exists(src_ingested_file):
        shutil.copy2(src_ingested_file, dst_ingested_file)


if __name__ == '__main__':
    store_model_into_pickle()
