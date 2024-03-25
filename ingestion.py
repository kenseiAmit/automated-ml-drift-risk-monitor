import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

############# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


############# Function for data ingestion
def merge_multiple_dataframe():
    """
    Check for datasets in input_folder_path, compile them together,
    remove duplicate rows, and write to finaldata.csv in output_folder_path.
    Also save the record of ingested files to ingestedfiles.txt.
    """
    with open('config.json', 'r') as config_file:
        current_config = json.load(config_file)
    in_path = current_config['input_folder_path']
    out_path = current_config['output_folder_path']

    os.makedirs(out_path, exist_ok=True)

    csv_files = [f for f in os.listdir(in_path) if f.endswith('.csv')]
    dfs = []
    ingested_records = []

    for file in sorted(csv_files):
        file_path = os.path.join(in_path, file)
        df = pd.read_csv(file_path)
        dfs.append(df)
        ingested_records.append(file)

    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        deduped_df = combined_df.drop_duplicates()
    else:
        deduped_df = pd.DataFrame()

    # Save deduplicated dataframe to finaldata.csv
    final_data_path = os.path.join(out_path, 'finaldata.csv')
    deduped_df.to_csv(final_data_path, index=False)

    # Save ingested file records to ingestedfiles.txt
    ingested_record_path = os.path.join(out_path, 'ingestedfiles.txt')
    with open(ingested_record_path, 'w') as f:
        for fname in ingested_records:
            f.write(f"{fname}\n")

    return deduped_df


if __name__ == '__main__':
    merge_multiple_dataframe()
