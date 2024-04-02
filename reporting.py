import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import diagnostics

############### Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path'])


############## Function for reporting
def score_model(output_filename='confusionmatrix.png'):
    """
    Calculate a confusion matrix using the test data and the deployed model
    and write the confusion matrix plot to output_model_path.
    """
    with open('config.json', 'r') as f:
        current_config = json.load(f)

    cur_test_path = current_config['test_data_path']
    cur_model_path = current_config['output_model_path']
    os.makedirs(cur_model_path, exist_ok=True)

    test_file = os.path.join(cur_test_path, 'testdata.csv')
    test_df = pd.read_csv(test_file)

    y_true = test_df['exited']
    y_pred = diagnostics.model_predictions(test_df)

    cm = metrics.confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Not Exited (0)', 'Exited (1)'],
                yticklabels=['Not Exited (0)', 'Exited (1)'])
    plt.title('Attrition Risk Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    out_file = os.path.join(cur_model_path, output_filename)
    plt.savefig(out_file)
    plt.close(fig)

    import shutil
    root_file = os.path.basename(output_filename)
    if os.path.abspath(out_file) != os.path.abspath(root_file):
        shutil.copy2(out_file, root_file)

    return cm


if __name__ == '__main__':
    score_model('confusionmatrix.png')
