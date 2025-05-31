import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
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
        'radius_gyration_ratio': 0.323  # Radius of gyration as % of segment length
    },
    'shank': {
        'mass_ratio': 0.0465,  # 4.65% of body mass
        'length_ratio': 0.246,  # 24.6% of height
        'com_ratio': 0.433,  # CoM at 43.3% from proximal end
        'radius_gyration_ratio': 0.302  # Radius of gyration
    },
    'foot': {
        'mass_ratio': 0.0145,  # 1.45% of body mass
        'length_ratio': 0.152,  # 15.2% of height
        'com_ratio': 0.500,  # CoM at 50% from proximal end
        'radius_gyration_ratio': 0.475  # Radius of gyration
    }
}

# Function to extract gait cycles (from your previous code)
def extract_gait_cycles(df, time_col, heel_col, min_time_gap=0.5):
    cycles = []
    start_indices = []
    heel_z = savgol_filter(df[heel_col].values, window_length=11, polyorder=2)
    print(f"{heel_col} stats: min={np.nanmin(heel_z):.2f}, max={np.nanmax(heel_z):.2f}, mean={np.nanmean(heel_z):.2f}")
    threshold = np.min(heel_z) + 0.2 * (np.max(heel_z) - np.min(heel_z))
    for i in range(1, len(heel_z) - 1):
        if (heel_z[i] < threshold) and (heel_z[i-1] > heel_z[i] < heel_z[i+1]):
            if not start_indices or (df[time_col].values[i] - df[time_col].values[start_indices[-1]]) >= min_time_gap:
                start_indices.append(i)
    print(f"Detected {len(start_indices)} heel strikes in {heel_col} at times {df[time_col].values[start_indices]}")
    for i in range(len(start_indices)-1):
        cycle_data = df.iloc[start_indices[i]:start_indices[i+1]].copy()
        if len(cycle_data) > 10:
            cycles.append(cycle_data)
    if len(start_indices) < 2:
        print(f"Warning: Insufficient heel strikes in {heel_col} to form a cycle.")
    return cycles

# Function to normalize a cycle to 0–100% (from your previous code)
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
                print(f"Warning: Failed to interpolate {col} in cycle (possible NaN or insufficient data)")
                norm_data[col] = np.zeros(n_points)
    return pd.DataFrame(norm_data)

# Function to calculate segment length from marker positions
def calculate_segment_length(cycle, proximal_marker, distal_marker):
    """Calculate segment length using Euclidean distance between proximal and distal markers."""
    if all(col in cycle.columns for col in [f'{proximal_marker}_X', f'{proximal_marker}_Y', f'{proximal_marker}_Z',
                                           f'{distal_marker}_X', f'{distal_marker}_Y', f'{distal_marker}_Z']):
        coords_prox = cycle[[f'{proximal_marker}_X', f'{proximal_marker}_Y', f'{proximal_marker}_Z']].values
        coords_dist = cycle[[f'{distal_marker}_X', f'{distal_marker}_Y', f'{distal_marker}_Z']].values
        lengths = np.sqrt(np.sum((coords_prox - coords_dist)**2, axis=1))
        return np.mean(lengths[pd.notna(lengths)])
    else:
        print(f"Markers {proximal_marker} or {distal_marker} not found in cycle.")
        return None

# Function to calculate CoM positions for a normalized cycle
def calculate_com_positions(cycle):
    """Calculate CoM positions for thigh, shank, and foot in a normalized cycle."""
    com_positions = {
        'thigh': np.zeros((len(cycle), 3)),
        'shank': np.zeros((len(cycle), 3)),
        'foot': np.zeros((len(cycle), 3))
    }

    for i in range(len(cycle)):
        # Thigh: Approximate hip as offset from RKNE
        if all(col in cycle.columns for col in ['RKNE_X', 'RKNE_Y', 'RKNE_Z']):
            knee_pos = np.array([cycle['RKNE_X'].iloc[i], cycle['RKNE_Y'].iloc[i], cycle['RKNE_Z'].iloc[i]])
            thigh_length = segment_params['thigh']['length_ratio'] * height
            com_distance = segment_params['thigh']['com_ratio'] * thigh_length
            hip_pos = knee_pos + np.array([0, 0, thigh_length])  # Simplified
            thigh_com = hip_pos - com_distance * (hip_pos - knee_pos) / thigh_length
            com_positions['thigh'][i] = thigh_com if not np.any(np.isnan(thigh_com)) else np.array([np.nan, np.nan, np.nan])
        else:
            com_positions['thigh'][i] = np.array([np.nan, np.nan, np.nan])

        # Shank: Between RKNE and RANK
        if all(col in cycle.columns for col in ['RKNE_X', 'RKNE_Y', 'RKNE_Z', 'RANK_X', 'RANK_Y', 'RANK_Z']):
            knee_pos = np.array([cycle['RKNE_X'].iloc[i], cycle['RKNE_Y'].iloc[i], cycle['RKNE_Z'].iloc[i]])
            ankle_pos = np.array([cycle['RANK_X'].iloc[i], cycle['RANK_Y'].iloc[i], cycle['RANK_Z'].iloc[i]])
            shank_length = np.linalg.norm(knee_pos - ankle_pos)
            com_distance = segment_params['shank']['com_ratio'] * shank_length
            shank_com = knee_pos - com_distance * (knee_pos - ankle_pos) / shank_length if shank_length > 0 else np.array([np.nan, np.nan, np.nan])
            com_positions['shank'][i] = shank_com if not np.any(np.isnan(shank_com)) else np.array([np.nan, np.nan, np.nan])
        else:
            com_positions['shank'][i] = np.array([np.nan, np.nan, np.nan])

        # Foot: Between RANK and RTOE
        if all(col in cycle.columns for col in ['RANK_X', 'RANK_Y', 'RANK_Z', 'RTOE_X', 'RTOE_Y', 'RTOE_Z']):
            ankle_pos = np.array([cycle['RANK_X'].iloc[i], cycle['RANK_Y'].iloc[i], cycle['RANK_Z'].iloc[i]])
            toe_pos = np.array([cycle['RTOE_X'].iloc[i], cycle['RTOE_Y'].iloc[i], cycle['RTOE_Z'].iloc[i]])
            foot_length = np.linalg.norm(ankle_pos - toe_pos)
            com_distance = segment_params['foot']['com_ratio'] * foot_length
            foot_com = ankle_pos - com_distance * (ankle_pos - toe_pos) / foot_length if foot_length > 0 else np.array([np.nan, np.nan, np.nan])
            com_positions['foot'][i] = foot_com if not np.any(np.isnan(foot_com)) else np.array([np.nan, np.nan, np.nan])
        else:
            com_positions['foot'][i] = np.array([np.nan, np.nan, np.nan])

    return com_positions

# Load and process CSV files
file_prefix = 'S13_D1_shoe_normal_rd1_'
file_numbers = range(16, 21)
dfs = []
position_cols = ['RTOE_X', 'RTOE_Y', 'RTOE_Z', 'RHEE_X', 'RHEE_Y', 'RHEE_Z',
                 'RKNE_X', 'RKNE_Y', 'RKNE_Z', 'RANK_X', 'RANK_Y', 'RANK_Z']

for num in file_numbers:
    filename = f'{file_prefix}{num}.csv'
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        # Convert position columns from mm to m
        for col in position_cols:
            if col in df.columns:
                df[col] = df[col] / 1000.0
        # Handle NaN by interpolating
        df[position_cols] = df[position_cols].interpolate(method='linear', limit_direction='both').fillna(method='ffill').fillna(method='bfill')
        dfs.append(df)
        print(f"Loaded {filename}")
    else:
        print(f"File {filename} not found.")

# Extract and normalize gait cycles
all_cycles = []
numeric_columns = ['time'] + position_cols
for df in dfs:
    if 'RHEE_Z' in df.columns:
        cycles = extract_gait_cycles(df, 'time', 'RHEE_Z', min_time_gap=0.5)
        for cycle in cycles:
            norm_cycle = normalize_cycle(cycle, numeric_columns)
            if not norm_cycle.empty and len(norm_cycle) == 101:
                all_cycles.append(norm_cycle)
        print(f"Extracted {len(cycles)} cycles from {filename}")

# Calculate segment lengths and CoM positions for each cycle
segment_lengths = {'thigh': [], 'shank': [], 'foot': []}
com_positions_all = {'thigh': [], 'shank': [], 'foot': []}

for cycle in all_cycles:
    # Calculate segment lengths
    shank_length = calculate_segment_length(cycle, 'RKNE', 'RANK')
    foot_length = calculate_segment_length(cycle, 'RANK', 'RTOE')
    if shank_length is not None:
        segment_lengths['shank'].append(shank_length)
    if foot_length is not None:
        segment_lengths['foot'].append(foot_length)
    segment_lengths['thigh'].append(segment_params['thigh']['length_ratio'] * height)

    # Calculate CoM positions
    com_positions = calculate_com_positions(cycle)
    for segment in com_positions:
        com_positions_all[segment].append(com_positions[segment])

# Average segment lengths
avg_segment_lengths = {
    'thigh': np.mean(segment_lengths['thigh']),
    'shank': np.mean(segment_lengths['shank']) if segment_lengths['shank'] else segment_params['shank']['length_ratio'] * height,
    'foot': np.mean(segment_lengths['foot']) if segment_lengths['foot'] else segment_params['foot']['length_ratio'] * height
}

# Calculate inertial parameters
inertial_params = {}
for segment in ['thigh', 'shank', 'foot']:
    mass = segment_params[segment]['mass_ratio'] * body_mass
    length = avg_segment_lengths[segment]
    com_distance = segment_params[segment]['com_ratio'] * length
    radius_gyration = segment_params[segment]['radius_gyration_ratio'] * length
    moment_inertia = mass * radius_gyration**2
    inertial_params[segment] = {
        'mass': mass,
        'length': length,
        'com_distance': com_distance,
        'moment_inertia': moment_inertia
    }

# Average CoM positions across cycles
avg_com_positions = {}
for segment in com_positions_all:
    valid_coms = [com for com in com_positions_all[segment] if not np.any(np.isnan(com))]
    if valid_coms:
        avg_com_positions[segment] = np.nanmean(valid_coms, axis=0)
    else:
        avg_com_positions[segment] = np.full((101, 3), np.nan)
        print(f"Warning: No valid CoM data for {segment}")

# Print inertial parameters
print("\nAverage Body Inertial Parameters:")
for segment, params in inertial_params.items():
    print(f"\n{segment.capitalize()}:")
    print(f"  Mass: {params['mass']:.3f} kg")
    print(f"  Length: {params['length']:.3f} m")
    print(f"  CoM Distance from Proximal End: {params['com_distance']:.3f} m")
    print(f"  Moment of Inertia: {params['moment_inertia']:.3f} kg·m²")

# Print sample CoM positions
print("\nAverage CoM Positions Over Gait Cycle (first 5 points):")
for segment in avg_com_positions:
    print(f"\n{segment.capitalize()} CoM (x, y, z):")
    for i in range(min(5, len(avg_com_positions[segment]))):
        print(f"  Gait Cycle {i*100/101:.1f}%: {avg_com_positions[segment][i]}")

# Plot CoM trajectories (Z-axis)
plt.figure(figsize=(12, 6))
percentage = np.linspace(0, 100, 101)
for segment in avg_com_positions:
    plt.plot(percentage, avg_com_positions[segment][:, 2], label=f'{segment.capitalize()} CoM Z')
plt.xlabel('Gait Cycle (%)')
plt.ylabel('CoM Position Z (m)')
plt.title('Average Center of Mass Trajectories (Z-axis) Over Gait Cycle')
plt.legend()
plt.grid()
plt.show()