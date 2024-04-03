import os
import json
import pickle
import time
import subprocess
import sys
import pandas as pd
import requests

import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting
import apicalls


def run_process():
    with open('config.json', 'r') as f:
        config = json.load(f)

    input_folder_path = config['input_folder_path']
    output_folder_path = config['output_folder_path']
    test_data_path = config['test_data_path']
    output_model_path = config['output_model_path']
    prod_deployment_path = config['prod_deployment_path']

    ################## Check and read new data
    # 1. Read ingestedfiles.txt from prod_deployment_path
    deployed_ingested_file = os.path.join(prod_deployment_path, 'ingestedfiles.txt')
    if os.path.exists(deployed_ingested_file):
        with open(deployed_ingested_file, 'r') as f:
            ingested_files = [line.strip() for line in f if line.strip()]
    else:
        ingested_files = []

    # 2. Check input_folder_path and identify files not in ingestedfiles.txt
    source_files = [f for f in os.listdir(input_folder_path) if f.endswith('.csv')]
    new_files = [f for f in source_files if f not in ingested_files]

    ################## Deciding whether to proceed, part 1
    # If no new files are found, stop before drift checking or reporting.
    if not new_files:
        print("No new data found. Stopping process.")
        return False

    print(f"Found new data files: {new_files}. Running ingestion...")
    ingestion.merge_multiple_dataframe()

    ################## Checking for model drift
    # Read the baseline latestscore.txt and model from prod_deployment_path
    baseline_score_file = os.path.join(prod_deployment_path, 'latestscore.txt')
    deployed_model_file = os.path.join(prod_deployment_path, 'trainedmodel.pkl')

    with open(baseline_score_file, 'r') as f:
        baseline_f1 = float(f.read().strip())

    with open(deployed_model_file, 'rb') as f:
        deployed_model = pickle.load(f)

    # Score deployed model on the newly ingested data (no score_path, do not overwrite baseline)
    new_data_path = os.path.join(output_folder_path, 'finaldata.csv')
    new_df = pd.read_csv(new_data_path)
    new_f1 = scoring.score_model(deployed_model, new_df)

    print(f"Baseline F1 score: {baseline_f1}")
    print(f"Score on newly ingested data: {new_f1}")

    model_drift = new_f1 < baseline_f1

    ################## Deciding whether to proceed, part 2
    if model_drift:
        print("Model drift detected! Retraining and redeploying model...")
        # 1. Train replacement model on latest ingested data
        training.train_model()

        # 2. Score replacement model on test data and save latestscore.txt in output_model_path
        with open(os.path.join(output_model_path, 'trainedmodel.pkl'), 'rb') as f:
            replacement_model = pickle.load(f)
        test_df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
        replacement_score_path = os.path.join(output_model_path, 'latestscore.txt')
        scoring.score_model(replacement_model, test_df, score_path=replacement_score_path)

        # 3. Redeploy replacement model, new score, and ingestedfiles.txt
        deployment.store_model_into_pickle()
        print("Replacement model deployed to production.")
    else:
        print("No model drift detected. Keeping deployed model unchanged.")

    ################## Diagnostics and reporting
    print("Generating updated confusion matrix (confusionmatrix2.png)...")
    reporting.score_model('confusionmatrix2.png')

    print("Calling API endpoints and saving responses (apireturns2.txt)...")
    api_process = None
    try:
        # Check if API server is responding
        requests.get(f"{apicalls.URL}/scoring", timeout=2)
    except Exception:
        # Start API server in background if not running
        print("Starting API server in background for apicalls...")
        api_process = subprocess.Popen([sys.executable, 'app.py'],
                                       stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
        # Wait up to 15 seconds for API server to become ready
        for _ in range(15):
            time.sleep(1)
            try:
                requests.get(f"{apicalls.URL}/scoring", timeout=1)
                break
            except Exception:
                pass

    try:
        apicalls.call_apis('apireturns2.txt')
        print("API responses successfully saved to apireturns2.txt")
    finally:
        if api_process is not None:
            api_process.terminate()
            api_process.wait()

    return True


if __name__ == '__main__':
    run_process()
