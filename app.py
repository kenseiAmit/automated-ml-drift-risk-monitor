from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
import scoring
import diagnostics

###################### Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])

prediction_model = None


####################### Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    """
    Load the dataset at the location supplied in the request into a DataFrame,
    then pass it to the Step 3 model_predictions function (deployed model).
    """
    if request.method == 'OPTIONS':
        return ('', 200)

    req_data = request.get_json(silent=True) or {}
    filepath = req_data.get('dataset_path') or req_data.get('filepath') or req_data.get('path')
    if not filepath:
        filepath = request.form.get('dataset_path') or request.args.get('dataset_path') or request.data.decode().strip()
    if not filepath:
        with open('config.json', 'r') as f:
            cfg = json.load(f)
        filepath = os.path.join(cfg['test_data_path'], 'testdata.csv')

    df = pd.read_csv(filepath)
    preds = diagnostics.model_predictions(df)
    return jsonify(preds), 200


####################### Scoring Endpoint
@app.route("/scoring", methods=['GET', 'OPTIONS'])
def scoring_endpoint():
    """
    Score the deployed model on test_data_path/testdata.csv using score_model.
    Do not overwrite the deployed baseline score.
    """
    if request.method == 'OPTIONS':
        return ('', 200)

    with open('config.json', 'r') as f:
        cfg = json.load(f)
    prod_path = cfg['prod_deployment_path']
    test_path = cfg['test_data_path']

    model_file = os.path.join(prod_path, 'trainedmodel.pkl')
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    test_file = os.path.join(test_path, 'testdata.csv')
    test_df = pd.read_csv(test_file)

    f1 = scoring.score_model(model, test_df)
    return jsonify(f1), 200


####################### Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def summarystats_endpoint():
    """
    Return means, medians, and standard deviations for numeric columns.
    """
    if request.method == 'OPTIONS':
        return ('', 200)

    stats = diagnostics.dataframe_summary()
    return jsonify(stats), 200


####################### Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnostics_endpoint():
    """
    Return ingestion/training timings, missing-value percentages for all columns,
    and the dependency version table. Do not replace the deployed model.
    """
    if request.method == 'OPTIONS':
        return ('', 200)

    timings = diagnostics.execution_time()
    missing = diagnostics.missing_data()
    deps = diagnostics.outdated_packages_list()

    return jsonify({
        'execution_time': timings,
        'missing_data': missing,
        'dependencies': deps
    }), 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
