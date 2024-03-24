import requests
import json
import os

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

with open('config.json', 'r') as f:
    config = json.load(f)

output_model_path = config['output_model_path']


def call_apis(output_filename='apireturns.txt'):
    with open('config.json', 'r') as f:
        cur_cfg = json.load(f)
    cur_output_dir = cur_cfg['output_model_path']
    os.makedirs(cur_output_dir, exist_ok=True)

    # Call each API endpoint and store the responses
    # 1. Prediction endpoint
    response1 = requests.post(f"{URL}/prediction", json={"dataset_path": "testdata/testdata.csv"})

    # 2. Scoring endpoint
    response2 = requests.get(f"{URL}/scoring")

    # 3. Summary statistics endpoint
    response3 = requests.get(f"{URL}/summarystats")

    # 4. Diagnostics endpoint
    response4 = requests.get(f"{URL}/diagnostics")

    # Combine all API responses
    responses = (
        "=== Model Predictions ===\n"
        f"Status: {response1.status_code}\n"
        f"Response:\n{response1.text}\n\n"
        "=== Model Score (F1) ===\n"
        f"Status: {response2.status_code}\n"
        f"Response:\n{response2.text}\n\n"
        "=== Summary Statistics ===\n"
        f"Status: {response3.status_code}\n"
        f"Response:\n{response3.text}\n\n"
        "=== Diagnostics ===\n"
        f"Status: {response4.status_code}\n"
        f"Response:\n{response4.text}\n"
    )

    # Write the responses to output_model_path
    output_path = os.path.join(cur_output_dir, output_filename)
    with open(output_path, 'w') as f:
        f.write(responses)

    # Also save to current directory for submission bundle convenience
    root_path = os.path.basename(output_filename)
    if os.path.abspath(output_path) != os.path.abspath(root_path):
        with open(root_path, 'w') as f:
            f.write(responses)

    return responses


if __name__ == '__main__':
    call_apis('apireturns.txt')
