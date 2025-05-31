import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.interpolate import interp1d
import os

# Subject parameters
body_mass = 62  # kg
height = 1.65  # m
g = 9.81  # m/s^2

# Dempster's anthropometric data (Winter, 2009)
segment_params = {
    'thigh': {
        'mass_ratio': 0.100,  # 10% of body mass
        'length_ratio': 0.245,  # 24.5% of height
        'com_ratio': 0.433,  # CoM at 43.3% from proximal end
        'radius_gyration_ratio': 0.323
    },
    'shank': {
        'mass_ratio': 0.0465,  # 4.65% of body mass
        'length_ratio': 0.246,  # 24.6% of height
        'com_ratio': 0.433,  # CoM at 43.3% from proximal end
        'radius_gyration_ratio': 0.302
    },
    'foot': {
        'mass_ratio': 0.0145,  # 1.45% of body mass
        'length_ratio': 0.152,  # 15.2% of height
        'com_ratio': 0.500,  # CoM at 50% from proximal end
        'radius_gyration_ratio': 0.475
    }
}

# Butterworth filter function
def butter_lowpass_filter(data, cutoff=6, fs=100, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Function to extract gait cycles
def extract_gait_cycles(df, time_col, heel_col, min_time_gap=0.5):
    cycles = []
    start_indices = []
    heel_z = savgol_filter(df[heel_col].values, window_length=11, polyorder=2)
    threshold = np.min(heel_z) + 0.2 * (np.max(heel_z) - np.min(heel_z))
    for i in range(1, len(heel_z) - 1):
        if (heel_z[i] < threshold) and (heel_z[i-1] > heel_z[i] < heel_z[i+1]):
            if not start_indices or (df[time_col].values[i] - df[time_col].values[start_indices[-1]]) >= min_time_gap:
                start_indices.append(i)
    for i in range(len(start_indices)-1):
        cycle_data = df.iloc[start_indices[i]:start_indices[i+1]].copy()
        if len(cycle_data) > 10:
            cycles.append(cycle_data)
    return cycles

# Function to normalize a cycle to 0–100%
def normalize_cycle(cycle, columns, n_points=101):
    time = cycle['time'].values
    cycle_percentage = np.linspace(0, 100, n_points)
    norm_data = {'percentage': cycle_percentage}
    for col in columns:
        if col in cycle.columns and col != 'time':
            try:
                interp_func = interp1d(np.linspace(0, 100, len(time)), cycle[col].values, kind='linear', fill_value="extrapolate")
                norm_data[col] = interp_func(cycle_percentage)
            except ValueError:
                norm_data[col] = np.zeros(n_points)
    return pd.DataFrame(norm_data)

# Function to select cycle with force plate strike
def select_force_plate_cycle(cycles, grf_cols):
    selected_cycle = None
    max_grf = 0
    for cycle in cycles:
        if all(col in cycle.columns for col in grf_cols):
            total_fz = cycle['FP1_6965 - Force_Fz'] + cycle['FP2_6966 - Force_Fz']
            if len(total_fz[total_fz > 10]) > 10:  # Ensure significant GRF
                peak_grf = total_fz.max()
                if peak_grf > max_grf:
                    max_grf = peak_grf
                    selected_cycle = cycle
    return selected_cycle

# Load and process CSV files
file_prefix = 'S13_D1_shoe_normal_rd1_'
file_numbers = range(16, 21)
dfs = []
position_cols = ['RTOE_X', 'RTOE_Y', 'RTOE_Z', 'RHEE_X', 'RHEE_Y', 'RHEE_Z',
                 'RKNE_X', 'RKNE_Y', 'RKNE_Z', 'RANK_X', 'RANK_Y', 'RANK_Z']
grf_cols = ['FP1_6965 - Force_Fx', 'FP1_6965 - Force_Fy', 'FP1_6965 - Force_Fz',
            'FP2_6966 - Force_Fx', 'FP2_6966 - Force_Fy', 'FP2_6966 - Force_Fz']
angle_cols = ['RKneeAngles_X', 'RAnkleAngles_X']
numeric_columns = ['time'] + position_cols + grf_cols + angle_cols

for num in file_numbers:
    filename = f'{file_prefix}{num}.csv'
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        # Convert positions to meters
        for col in position_cols:
            if col in df.columns:
                df[col] = df[col] / 1000.0
        # Interpolate NaN
        df[numeric_columns] = df[numeric_columns].interpolate(method='linear', limit_direction='both').fillna(0)
        dfs.append(df)
        print(f"Loaded {filename}")
    else:
        print(f"File {filename} not found.")

# Extract and normalize force plate cycles
selected_cycles = []
for df in dfs:
    if 'RHEE_Z' in df.columns:
        cycles = extract_gait_cycles(df, 'time', 'RHEE_Z', min_time_gap=0.5)
        force_plate_cycle = select_force_plate_cycle(cycles, grf_cols)
        if force_plate_cycle is not None:
            norm_cycle = normalize_cycle(force_plate_cycle, numeric_columns)
            if not norm_cycle.empty and len(norm_cycle) == 101:
                selected_cycles.append(norm_cycle)
                print(f"Selected force plate cycle from {filename}")
        else:
            print(f"No valid force plate cycle found in {filename}")

# Calculate inertial parameters
segment_lengths = {'thigh': [], 'shank': [], 'foot': []}
for cycle in selected_cycles:
    shank_length = np.mean(np.sqrt((cycle['RKNE_X'] - cycle['RANK_X'])**2 + (cycle['RKNE_Y'] - cycle['RANK_Y'])**2 + (cycle['RKNE_Z'] - cycle['RANK_Z'])**2))
    foot_length = np.mean(np.sqrt((cycle['RANK_X'] - cycle['RTOE_X'])**2 + (cycle['RANK_Y'] - cycle['RTOE_Y'])**2 + (cycle['RANK_Z'] - cycle['RTOE_Z'])**2))
    if not np.isnan(shank_length):
        segment_lengths['shank'].append(shank_length)
    if not np.isnan(foot_length):
        segment_lengths['foot'].append(foot_length)
    segment_lengths['thigh'].append(segment_params['thigh']['length_ratio'] * height)

avg_segment_lengths = {
    'thigh': np.mean(segment_lengths['thigh']),
    'shank': np.mean(segment_lengths['shank']) if segment_lengths['shank'] else segment_params['shank']['length_ratio'] * height,
    'foot': np.mean(segment_lengths['foot']) if segment_lengths['foot'] else segment_params['foot']['length_ratio'] * height
}

inertial_params = {}
for segment in ['thigh', 'shank', 'foot']:
    mass = segment_params[segment]['mass_ratio'] * body_mass
    length = avg_segment_lengths[segment]
    com_distance = segment_params[segment]['com_ratio'] * length
    inertial_params[segment] = {'mass': mass, 'length': length, 'com_distance': com_distance}

# Calculate joint reaction forces for selected cycles
reaction_forces = {
    'ankle_fx': [], 'ankle_fz': [],
    'knee_fx': [], 'knee_fz': [],
    'hip_fx': [], 'hip_fz': []
}
compressive_forces = {'ankle': [], 'knee': [], 'hip': []}
shear_forces = {'ankle': [], 'knee': [], 'hip': []}
fs = 100  # Sampling frequency (Hz)

for cycle in selected_cycles:
    cycle_reaction = {
        'ankle_fx': np.zeros(101), 'ankle_fz': np.zeros(101),
        'knee_fx': np.zeros(101), 'knee_fz': np.zeros(101),
        'hip_fx': np.zeros(101), 'hip_fz': np.zeros(101)
    }
    cycle_compressive = {'ankle': np.zeros(101), 'knee': np.zeros(101), 'hip': np.zeros(101)}
    cycle_shear = {'ankle': np.zeros(101), 'knee': np.zeros(101), 'hip': np.zeros(101)}
    dt = 1 / fs

    for i in range(2, 101):
        # Skip if NaN in critical data
        required_cols = ['RANK_X', 'RANK_Z', 'RKNE_X', 'RKNE_Z', 'RTOE_X', 'RTOE_Z'] + grf_cols
        if cycle[required_cols].iloc[i].isna().any():
            continue

        # GRF (sagittal plane: X, Z)
        grf = np.array([
            cycle['FP1_6965 - Force_Fx'].iloc[i] + cycle['FP2_6966 - Force_Fx'].iloc[i],
            cycle['FP1_6965 - Force_Fz'].iloc[i] + cycle['FP2_6966 - Force_Fz'].iloc[i]
        ])
        if grf[1] < 10:  # Skip low GRF
            continue

        # Foot segment
        ankle_pos = np.array([cycle['RANK_X'].iloc[i], cycle['RANK_Z'].iloc[i]])
        ankle_acc = np.array([
            (cycle['RANK_X'].iloc[i] - 2*cycle['RANK_X'].iloc[i-1] + cycle['RANK_X'].iloc[i-2]) / dt**2,
            (cycle['RANK_Z'].iloc[i] - 2*cycle['RANK_Z'].iloc[i-1] + cycle['RANK_Z'].iloc[i-2]) / dt**2
        ])
        ankle_acc = np.clip(ankle_acc, -1e3, 1e3)
        foot_mass = inertial_params['foot']['mass']
        foot_weight = np.array([0, -foot_mass * g])
        ankle_reaction = grf - foot_weight - foot_mass * ankle_acc
        cycle_reaction['ankle_fx'][i] = ankle_reaction[0]
        cycle_reaction['ankle_fz'][i] = ankle_reaction[1]

        # Shank segment
        knee_pos = np.array([cycle['RKNE_X'].iloc[i], cycle['RKNE_Z'].iloc[i]])
        knee_acc = np.array([
            (cycle['RKNE_X'].iloc[i] - 2*cycle['RKNE_X'].iloc[i-1] + cycle['RKNE_X'].iloc[i-2]) / dt**2,
            (cycle['RKNE_Z'].iloc[i] - 2*cycle['RKNE_Z'].iloc[i-1] + cycle['RKNE_Z'].iloc[i-2]) / dt**2
        ])
        knee_acc = np.clip(knee_acc, -1e3, 1e3)
        shank_mass = inertial_params['shank']['mass']
        shank_weight = np.array([0, -shank_mass * g])
        knee_reaction = ankle_reaction - shank_weight - shank_mass * knee_acc
        cycle_reaction['knee_fx'][i] = knee_reaction[0]
        cycle_reaction['knee_fz'][i] = knee_reaction[1]

        # Thigh segment
        thigh_mass = inertial_params['thigh']['mass']
        thigh_weight = np.array([0, -thigh_mass * g])
        hip_reaction = knee_reaction - thigh_weight
        cycle_reaction['hip_fx'][i] = hip_reaction[0]
        cycle_reaction['hip_fz'][i] = hip_reaction[1]

        # Compressive and shear forces
        # Ankle
        if 'RAnkleAngles_X' in cycle.columns and not np.isnan(cycle['RAnkleAngles_X'].iloc[i]):
            ankle_angle = np.radians(cycle['RAnkleAngles_X'].iloc[i])
            foot_axis = np.array([-np.sin(ankle_angle), np.cos(ankle_angle)])
            compressive = np.dot(ankle_reaction, foot_axis)
            shear = np.dot(ankle_reaction, np.array([np.cos(ankle_angle), np.sin(ankle_angle)]))
            cycle_compressive['ankle'][i] = compressive
            cycle_shear['ankle'][i] = shear

        # Knee
        if 'RKneeAngles_X' in cycle.columns and not np.isnan(cycle['RKneeAngles_X'].iloc[i]):
            knee_angle = np.radians(cycle['RKneeAngles_X'].iloc[i])
            shank_axis = np.array([-np.sin(knee_angle), np.cos(knee_angle)])
            compressive = np.dot(knee_reaction, shank_axis)
            shear = np.dot(knee_reaction, np.array([np.cos(knee_angle), np.sin(knee_angle)]))
            cycle_compressive['knee'][i] = compressive
            cycle_shear['knee'][i] = shear

        # Hip
        thigh_axis = np.array([0, 1])  # Vertical approximation
        compressive = np.dot(hip_reaction, thigh_axis)
        shear = np.dot(hip_reaction, np.array([1, 0]))
        cycle_compressive['hip'][i] = compressive
        cycle_shear['hip'][i] = shear

    # Filter reaction forces
    for key in cycle_reaction:
        if np.sum(np.abs(cycle_reaction[key])) > 0:
            cycle_reaction[key] = butter_lowpass_filter(cycle_reaction[key], cutoff=6, fs=fs)
    for key in cycle_compressive:
        if np.sum(np.abs(cycle_compressive[key])) > 0:
            cycle_compressive[key] = butter_lowpass_filter(cycle_compressive[key], cutoff=6, fs=fs)
        if np.sum(np.abs(cycle_shear[key])) > 0:
            cycle_shear[key] = butter_lowpass_filter(cycle_shear[key], cutoff=6, fs=fs)

    for key in reaction_forces:
        reaction_forces[key].append(cycle_reaction[key])
    for key in compressive_forces:
        compressive_forces[key].append(cycle_compressive[key])
        shear_forces[key].append(cycle_shear[key])

# Average across cycles
avg_reaction_forces = {key: np.nanmean(reaction_forces[key], axis=0) for key in reaction_forces}
avg_compressive_forces = {key: np.nanmean(compressive_forces[key], axis=0) for key in compressive_forces}
avg_shear_forces = {key: np.nanmean(shear_forces[key], axis=0) for key in shear_forces}

# Print statistics
print("\nAverage Joint Reaction Force Statistics (N):")
for joint in ['ankle', 'knee', 'hip']:
    for comp in ['fx', 'fz']:
        key = f'{joint}_{comp}'
        valid_data = avg_reaction_forces[key][~np.isnan(avg_reaction_forces[key])]
        if len(valid_data) > 0:
            print(f"{joint.capitalize()} {comp.upper()}: Max={valid_data.max():.2f} N, Min={valid_data.min():.2f} N, Mean={valid_data.mean():.2f} N, Std={valid_data.std():.2f} N")

print("\nAverage Compressive Force Statistics:")
for joint in ['ankle', 'knee', 'hip']:
    valid_data = avg_compressive_forces[joint][~np.isnan(avg_compressive_forces[joint])]
    if len(valid_data) > 0:
        print(f"{joint.capitalize()}: Max={valid_data.max():.2f} N, Min={valid_data.min():.2f} N, Mean={valid_data.mean():.2f} N, Std={valid_data.std():.2f} N")

print("\nAverage Shear Force Statistics:")
for joint in ['ankle', 'knee', 'hip']:
    valid_data = avg_shear_forces[joint][~np.isnan(avg_shear_forces[joint])]
    if len(valid_data) > 0:
        print(f"{joint.capitalize()}: Max={valid_data.max():.2f} N, Min={valid_data.min():.2f} N, Mean={valid_data.mean():.2f} N, Std={valid_data.std():.2f} N")

# Plot results
percentage = np.linspace(0, 100, 101)
plt.figure(figsize=(12, 6))
plt.plot(percentage, avg_reaction_forces['ankle_fz'], label='Ankle Fz')
plt.plot(percentage, avg_reaction_forces['knee_fz'], label='Knee Fz')
plt.plot(percentage, avg_reaction_forces['hip_fz'], label='Hip Fz')
plt.xlabel('Gait Cycle (%)')
plt.ylabel('Force (N)')
plt.title('Average Joint Reaction Forces (Vertical, Sagittal Plane)')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(percentage, avg_compressive_forces['ankle'], label='Ankle Compressive')
plt.plot(percentage, avg_compressive_forces['knee'], label='Knee Compressive')
plt.plot(percentage, avg_compressive_forces['hip'], label='Hip Compressive')
plt.xlabel('Gait Cycle (%)')
plt.ylabel('Force (N)')
plt.title('Average Compressive Forces')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(percentage, avg_shear_forces['ankle'], label='Ankle Shear')
plt.plot(percentage, avg_shear_forces['knee'], label='Knee Shear')
plt.plot(percentage, avg_shear_forces['hip'], label='Hip Shear')
plt.xlabel('Gait Cycle (%)')
plt.ylabel('Force (N)')
plt.title('Average Shear Forces')
plt.legend()
plt.grid()
plt.show()