"""
Author:  Aleksandra Feshchenko
Date:    05/05/2023

This script takes in raw files from data/raw/<experiment name>/<group X> and 
prepares them for the analysis by pulling Time,X,Y columns into a csv files
stored in the data/processed/<experiment name>/<group X>.

process.py  prepares data downloaded from Ethovision for analyze.py

"""

import os
import re
import pandas as pd
from pathlib import Path

#EXPERIMENT_NAME = "overexpression_oft_no_smooth_no_filter"
EXPERIMENT_NAME = "knockout_oft_no_smooth_no_filter"
#EXPERIMENT_NAME = "knockout_oft_smoothed"
#EXPERIMENT_NAME = "overexpression_oft_smoothed"

raw_folder = os.path.join('data', 'raw', EXPERIMENT_NAME)
preprocessed_folder = os.path.join('data', 'preprocessed', EXPERIMENT_NAME)
processed_folder = os.path.join('data', 'processed', EXPERIMENT_NAME)

# Create pre/processed experiment folder if doesn't exist
Path(preprocessed_folder).mkdir(parents=True, exist_ok=True)
Path(processed_folder).mkdir(parents=True, exist_ok=True)

# Find groups
groups = []
with os.scandir(raw_folder) as it:
    for group in it:
        if group.is_dir():
            groups.append(group.name)

print(f'Found {len(groups)} groups: {groups}\n')

print("""
###################
Raw -> Preprocessed
###################
""")

# Preprocess files. Convert from excel to csv
for group in groups:
    raw_group_folder = os.path.join(raw_folder, group)
    print(f'Looking through group folder {raw_group_folder}')

    preprocessed_group_folder = os.path.join(preprocessed_folder, group)
    preprocessed_head_folder = os.path.join(preprocessed_group_folder, 'head')
    preprocessed_body_folder = os.path.join(preprocessed_group_folder, 'body')
    Path(preprocessed_head_folder).mkdir(parents=True, exist_ok=True)
    Path(preprocessed_body_folder).mkdir(parents=True, exist_ok=True)

    with os.scandir(raw_group_folder) as it:
        for file in it:
            if file.is_file() and file.name.endswith('.xlsx'):

                filename_base = file.name.replace('.xlsx', '')
                preprocessed_head_path = os.path.join(preprocessed_head_folder, filename_base + '_head.csv')
                preprocessed_body_path = os.path.join(preprocessed_body_folder, filename_base + '_body.csv')

                if os.path.exists(preprocessed_head_path) and os.path.exists(preprocessed_body_path):
                    print(f'\tFound processeed `{file.name}`. Skipping.')
                    continue
                else:
                    print(f'\tPreprocessing file `{file.name}`')

                data = pd.read_excel(file.path, na_values=['-'])
                units_row_index = data[(data[data.columns[0]] == '(s)')].index[0]

                # HEADER
                header_data = data.head(units_row_index - 2)
                header_data = pd.Series(
                    header_data[header_data.columns[1]].tolist(),
                    index=header_data[header_data.columns[0]]
                )
                header_data['file_number'] = filename_base
                header_data.to_csv(preprocessed_head_path)
                
                # BODY
                body_data_raw = pd.DataFrame(
                    data.iloc[units_row_index + 1:].values,
                    columns = data.iloc[units_row_index - 1],
                )
                body_data_raw.to_csv(preprocessed_body_path, index=False)


print("""
#########################
Preprocessed -> Processed
#########################
""")

# Preprocessed to processed
for group in groups:
    preprocessed_group_folder = os.path.join(preprocessed_folder, group)
    # preprocessed_group_head_folder = os.path.join(preprocessed_group_folder, 'head')
    preprocessed_group_body_folder = os.path.join(preprocessed_group_folder, 'body')

    processed_group_folder = os.path.join(processed_folder, group)
    # processed_group_metadata_folder = os.path.join(processed_group_folder, 'metadata')
    processed_group_trials_folder = os.path.join(processed_group_folder, 'timed_coordinates')
    # Path(processed_group_metadata_folder).mkdir(parents=True, exist_ok=True)
    Path(processed_group_trials_folder).mkdir(parents=True, exist_ok=True)

    print(f'Looking through group folder {preprocessed_group_body_folder}')
    with os.scandir(preprocessed_group_body_folder) as it:
        for file in it:
            if file.is_file() and file.name.endswith('.csv'):
                # if os.path.exists(preprocessed_head_path) and os.path.exists(preprocessed_body_path):
                #     print(f'\tFound processeed `{file.name}`. Skipping.')
                #     continue
                # else:
                    # print(f'\tProcessing file `{file.name}`')
                print(f'\tProcessing file `{file.name}`')

                data = pd.read_csv(file.path)
                data = data.rename(columns={'Trial time': 'time', 'X center': 'x_center', 'Y center': 'y_center'})
                data = data[['time', 'x_center', 'y_center']]

                processed_filename = os.path.join(
                    processed_group_trials_folder, 
                    # re.search(r"\d+\_body\.csv$", file.name).group().replace('_body', '')
                    file.name.replace('_body', '')
                )
                print(f'\t Saving to `{processed_filename}`')
                data.to_csv(processed_filename, index=False)

