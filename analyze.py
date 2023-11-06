"""
Author:  Aleksandra Feshchenko
Date:    05/05/2023

Performs bout finding on experiment files located in processed folder such as
    data/processed/<experiment name>/<group X>/timed_coordinates


SMOOTHING_HALF_WINDOW_SIZE defines how many points before and after measurement to include in smoothing
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#EXPERIMENT_NAME = "overexpression_oft_no_smooth_no_filter"
EXPERIMENT_NAME = "knockout_oft_no_smooth_no_filter"
#EXPERIMENT_NAME = "knockout_oft_smoothed"
#EXPERIMENT_NAME = "overexpression_oft_smoothed"

#===============

#EXPERIMENT_NAME = "juv"
#EXPERIMENT_NAME = "adult"
#EXPERIMENT_NAME = "aged"

#================


TIME_WINDOW_START_SECONDS = 20 * 60  # Start of time window
TIME_WINDOW_END_SECONDS = 30 * 60  # End of time window
#SMOOTHING_HALF_WINDOW_SIZE = 0 
SMOOTHING_HALF_WINDOW_SIZE = 8  # 0 means no smoothing
BOUT_START_VELOCITY_THRESHOLD = 2.5  # cm/s
BOUT_END_VELOCITY_THRESHOLD = 1.5  # cm/s
BOUT_DURATION_THRESHOLD = 0.5  # s
NO_ACC_THRESHOLD = 1.0 # cm/s^2 - no change of velocity zone
SMOOTH_CONFIG = f'hw_{SMOOTHING_HALF_WINDOW_SIZE}_vstart_{BOUT_START_VELOCITY_THRESHOLD}_vend_{BOUT_END_VELOCITY_THRESHOLD}_bdur_{BOUT_DURATION_THRESHOLD}'
CONFIG = f'{SMOOTH_CONFIG}_noacc_{NO_ACC_THRESHOLD}_tstart_{TIME_WINDOW_START_SECONDS}_tend_{TIME_WINDOW_END_SECONDS}'

# If True, the code will only analyze specified files and plot them without overriding group stat tables
# If False, the code will analyze all trials and generate group stat tables, then plot all the trials
PLOT_TRIALS = False

# Only used if PLOT_TRIALS is True
PROCESSED_TRIALS_TO_PLOT = [
    "Raw data-Master SmoLL_ChATCRE-Trial     8.csv", 
    "Raw data-Master SmoLL_ChATCRE-Trial    39.csv", 
    "Raw data-Master SmoLL_ChATCRE-Trial    23.csv",
    "Raw data-Master SmoLL_ChATCRE-Trial    42.csv"
]

if PLOT_TRIALS:
    print('PLOT_TRIALS = True  | Only Plotting specified files')

processed_dir = f'data/processed/{EXPERIMENT_NAME}/'

assert TIME_WINDOW_START_SECONDS < TIME_WINDOW_END_SECONDS

# Caches smoothening and new column calculated data
smoothed_dir = f'data/smoothed/{EXPERIMENT_NAME}/{SMOOTH_CONFIG}'
Path(smoothed_dir).mkdir(parents=True, exist_ok=True)


def quadratic_smooth(x):
    """
    Smoothes with a quadratic polynomial
    """
    mean = np.mean(x)
    # fit a quadratic polynomial to the data
    p = np.polyfit(np.arange(len(x)) - len(x) / 2, x - mean, 2)
    # evaluate the polynomial at the center point
    return np.polyval(p, 0) + mean


def smooth_data(data: pd.DataFrame, smoothed_filepath: str):
    """
    Takes in data from processed folder with time, x_center and y_center

    Adds new columns:
        x_smooth
        y_smooth
        delta_x
        delta_y
        delta_t
        distance
        velocity
        acceleration
    """
    if Path(smoothed_filepath).is_file():
        print(f'\t- Found smoothed `{smoothed_filepath}`')
        data = pd.read_csv(smoothed_filepath)
    else:
        if SMOOTHING_HALF_WINDOW_SIZE == 0:
            # No smoothing, just copy over values
            data['x_smooth'] = data['x_center']
            data['y_smooth'] = data['y_center']
        else:
            data['x_smooth'] = data['x_center'].rolling(window=2 * SMOOTHING_HALF_WINDOW_SIZE + 1, center=True).apply(quadratic_smooth)
            data['y_smooth'] = data['y_center'].rolling(window=2 * SMOOTHING_HALF_WINDOW_SIZE + 1, center=True).apply(quadratic_smooth)

        data['delta_x'] = data['x_smooth'] - data['x_smooth'].shift(1)
        data['delta_y'] = data['y_smooth'] - data['y_smooth'].shift(1)
        data['delta_t'] = data['time'] - data['time'].shift(1)
        data['distance'] = np.sqrt(data['delta_x'] ** 2 + data['delta_y'] ** 2)
        data['velocity'] = data['distance'] / data['delta_t']
        data['acceleration'] = data['velocity'] - data['velocity'].shift(1)
        data.to_csv(smoothed_filepath, index=False)
    return data


def apply_min_diff(arr, min_diff):
    """
    arr = np.array([1, 3, 2, 1, 2, 3, 5, 6, 5, 6, 4, -1, 2])

    min_diff = 3.5

    produces [1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, -1, -1]
    """
    if len(arr) == 0:
        return np.array([])

    new_arr = [arr[0]]
    for i in range(1, len(arr)):
        diff = abs(arr[i] - new_arr[-1])
        if diff > min_diff:
            new_arr.append(arr[i])
        else:
            new_arr.append(new_arr[-1])
    return np.array(new_arr)


def compute_bouts(data: pd.DataFrame):
    """
    Adds bout info to data and returns bouts info
    """
    data['in_bout'] = False

    bouts = []
    bout_start = None
    bout_end = None
    for _, row in data.iterrows():
        # Check for bout start condition
        if row['velocity'] > BOUT_START_VELOCITY_THRESHOLD and bout_start is None:
            bout_start = row['time']
        # Check for bout end condition
        elif row['velocity'] < BOUT_END_VELOCITY_THRESHOLD and bout_start is not None:
            bout_end = row['time']
            bout_mask = (data['time'] >= bout_start) & (data['time'] < bout_end)
            velocities = data.loc[bout_mask, 'velocity']
            accelerations = data.loc[bout_mask, 'acceleration']

            velocities_w_min_diff = apply_min_diff(velocities.values, NO_ACC_THRESHOLD)
            velocities_diff_sign = np.sign(np.diff(velocities_w_min_diff))
            acc_signs = np.sum(np.diff(velocities_diff_sign[velocities_diff_sign != 0]) != 0)

            max_vel_time = data.loc[bout_mask].time.iloc[velocities.argmax()]
            max_acc_time = data.loc[bout_mask].time.iloc[accelerations.argmax()]

            first_peak = (accelerations < 0).idxmax()
            last_peak = (accelerations[::-1] >= 0).idxmax()
            time_to_first_vel_peak = data.at[first_peak, 'time'] - bout_start
            time_from_last_vel_peak_to_end = bout_end - data.at[last_peak, 'time']

            ten_pct_elems = int(np.ceil(len(velocities) / 10))

            bouts.append({
                'time_start': bout_start,
                'time_end': bout_end,
                'duration': bout_end - bout_start,  
                'velocity_mean': velocities.mean(),
                'velocity_max': velocities.max(),
                'distance': velocities.mean() * (bout_end - bout_start),
                'acceleration_mean': accelerations.mean(),
                'acceleration_max': accelerations.max(),
                'acceleration_min': accelerations.min(),
                'acc_changes': acc_signs,
                'acc_changes_per_s': acc_signs / (bout_end - bout_start),
                'bout_pct_time_till_max_vel': (max_vel_time - bout_start) / (bout_end - bout_start),
                'bout_pct_time_till_max_acc': (max_acc_time - bout_start) / (bout_end - bout_start),
                'max_vel_first_10pct': velocities[:ten_pct_elems].max(),
                'max_acc_first_10pct': accelerations[:ten_pct_elems].max(),
                'max_vel_last_10pct': velocities[-ten_pct_elems:].max(),
                'max_acc_last_10pct': accelerations[-ten_pct_elems:].max(),
                'time_to_first_vel_peak': time_to_first_vel_peak,
                'time_from_last_vel_peak_to_end': time_from_last_vel_peak_to_end,
                'time_to_first_vel_peak_pct': time_to_first_vel_peak / (bout_end - bout_start),
                'time_from_last_vel_peak_to_end_pct': time_from_last_vel_peak_to_end / (bout_end - bout_start),

            })
            data.loc[bout_mask, 'in_bout'] = True
            bout_start = None
            bout_end = None
    
    if len(bouts) == 0:
        bouts = pd.DataFrame([], columns=[
            'time_start',
            'time_end',
            'duration',
            'velocity_mean',
            'velocity_max',
            'distance',
            'acceleration_mean',
            'acceleration_max',
            'acceleration_min',
            'acc_changes',
            'acc_changes_per_s',
            'bout_pct_time_till_max_vel',
            'bout_pct_time_till_max_acc',
            'max_vel_first_10pct',
            'max_acc_first_10pct',
            'max_vel_last_10pct',
            'max_acc_last_10pct',
            'time_to_first_vel_peak',
            'time_from_last_vel_peak_to_end',
            'time_to_first_vel_peak_pct',
            'time_from_last_vel_peak_to_end_pct',
        ])
    else:

        bouts = pd.DataFrame(bouts).round(3)
    return data, bouts



# ================================================================================
# ============================ Bout Finder =======================================
# ================================================================================
print('Looking in processed_dir', processed_dir)

for group in os.listdir(processed_dir):
    if group.startswith('.'):
        continue  # ignore filesystem files
    
    print(f'Processing group : {group}')

    # Temporary cache folder to store smoothened data since smoothening takes a long time.
    smoothed_group_folder = os.path.join(smoothed_dir, group)
    Path(smoothed_group_folder).mkdir(parents=True, exist_ok=True)

    # Final result folder
    analyzed_group_folder = os.path.join('data', 'analyzed', EXPERIMENT_NAME, CONFIG, group)
    Path(analyzed_group_folder).mkdir(parents=True, exist_ok=True)

    # A single table summarizing all trial statistics within the group
    analyzed_group_stats_path = os.path.join(analyzed_group_folder, 'group_stats.csv')
    group_stats = []

    trial_files = os.listdir(os.path.join(processed_dir, group, 'timed_coordinates'))
    trial_files = [file for file in trial_files if not file.startswith('.')]  # Ignores any hidden files

    # Iterate through each group trial
    for trial_num, trial_file in enumerate(trial_files):
        if PLOT_TRIALS and (trial_file not in PROCESSED_TRIALS_TO_PLOT):
            print(f'skipping {trial_file=}')
            continue

        analyzed_trial_bouts_path = os.path.join(analyzed_group_folder, trial_file).replace('.csv', '_bouts.csv')

        print(f'\ttrial : {trial_file}')
        processed_filepath = os.path.join(processed_dir, group, 'timed_coordinates', trial_file)
        smoothed_filepath = os.path.join(smoothed_group_folder, trial_file)

        trial = pd.read_csv(processed_filepath)
        trial = smooth_data(trial, smoothed_filepath)
        trial = trial[(TIME_WINDOW_START_SECONDS <= trial.time) & (trial.time <= TIME_WINDOW_END_SECONDS)]
        trial = trial.dropna(axis=0)
        trial, bouts = compute_bouts(trial)

        print(f'len of bouts', len(bouts))

        micro_bouts = bouts[bouts['duration'] < BOUT_DURATION_THRESHOLD]
        bouts = bouts[bouts['duration'] >= BOUT_DURATION_THRESHOLD]
        bouts.to_csv(analyzed_trial_bouts_path, index=False)

        # Bout stats are only calculated for valid bouts. micro bouts are excluded
        group_stats.append({
            'trial': trial_file.replace('.csv', ''),
            'total_time': trial['time'].max() - trial['time'].min(),
            'total_trial_distance': trial['distance'].sum(),
            'valid_bouts': len(bouts),
            'excluded_micro_bouts': len(micro_bouts),
            'total_time_valid_bouts': bouts['duration'].sum(),
            'total_time_micro_bouts': micro_bouts['duration'].sum(),
            'total_time_outside_bouts': trial.loc[~trial['in_bout'], 'delta_t'].sum(),
            'total_distance_valid_bouts': bouts['distance'].sum(),
            'mean_velocity': trial['velocity'].mean(),
            'mean_bout_velocity': bouts['velocity_mean'].mean(),  # trial.loc[trial['in_bout'], 'velocity'].mean(),
            'mean_bout_duration': bouts['duration'].mean(),
            'mean_bout_distance': bouts['distance'].mean(),
            'max_bout_velocity': bouts['velocity_max'].max(),
            'max_bout_duration': bouts['duration'].max(),
            'max_bout_acceleration': bouts['acceleration_max'].max(),

            'acc_changes_mean': bouts['acc_changes'].mean(),
            'acc_changes_per_s_mean': bouts['acc_changes_per_s'].mean(),
            'bout_pct_time_till_max_vel_mean': bouts['bout_pct_time_till_max_vel'].mean(),
            'bout_pct_time_till_max_acc_mean': bouts['bout_pct_time_till_max_acc'].mean(),
            'bout_pct_time_after_max_vel_mean': 1 - bouts['bout_pct_time_till_max_vel'].mean(),
            'bout_pct_time_after_max_acc_mean': 1 - bouts['bout_pct_time_till_max_acc'].mean(),
            'max_vel_first_10pct_mean': bouts['max_vel_first_10pct'].mean(),
            'max_acc_first_10pct_mean': bouts['max_acc_first_10pct'].mean(),
            'max_vel_last_10pct_mean': bouts['max_vel_last_10pct'].mean(),
            'max_acc_last_10pct_mean': bouts['max_acc_last_10pct'].mean(),
            
            'time_to_first_vel_peak': bouts['time_to_first_vel_peak'].mean(),
            'time_from_last_vel_peak_to_end':bouts['time_from_last_vel_peak_to_end'].mean(),
            'time_to_first_vel_peak_pct': bouts['time_to_first_vel_peak_pct'].mean(),
            'time_from_last_vel_peak_to_end_pct': bouts['time_from_last_vel_peak_to_end_pct'].mean()
        })

        # ================================================================================
        # ============================== Plotting ========================================
        # ================================================================================

        if not PLOT_TRIALS:
            # Skip plotting
            continue

        fig, axis = plt.subplots(nrows=2, sharex=True)
        exp_title_name = EXPERIMENT_NAME.split('_')[0]
        fig.suptitle(f'{exp_title_name} - {group} - {trial_file}')
        ax1, ax2 = axis  # ax1 velocity, ax2 acceleration plots

        for start, end in zip(bouts['time_start'], bouts['time_end']):
            ax1.axvspan(start, end, color='green', alpha=0.2)

        ax1.plot(trial['time'], trial['velocity'], label='smoothed')
        ax1.axhline(BOUT_START_VELOCITY_THRESHOLD, ls='--', color='purple', label='Bout start threshold : {BOUT_START_VELOCITY_THRESHOLD} cm/s')
        ax1.axhline(BOUT_END_VELOCITY_THRESHOLD, ls='--', color='orange', label='Bout end threshold : {BOUT_END_VELOCITY_THRESHOLD} cm/s')
        ax1.set_ylabel('velocity')
        ax1.set_ylim([-1, 51])
        
        ax2.plot(trial['time'], trial['acceleration'], marker='o', alpha=0.2)
        ax2.axhline(0.0, ls='-', color='black')
        ax2.axhline(-NO_ACC_THRESHOLD, ls='--', color='orange')
        ax2.axhline(NO_ACC_THRESHOLD, ls='--', color='orange')
        ax2.set_ylabel('acc')
        ax2.set_xlabel('time [s]')
        # plt.show()
        
    if PLOT_TRIALS:
        # Do not override group stats if plotting
        continue

    group_stats = pd.DataFrame(group_stats).round(3)
    group_stats.to_csv(analyzed_group_stats_path, index=False)


if PLOT_TRIALS:
    plt.show()
