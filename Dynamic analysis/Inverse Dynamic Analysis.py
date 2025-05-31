import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Set backend for non-Jupyter environments
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.interpolate import interp1d
import os

# Anthropometric parameters (Winter, 2009)
body_mass = 62  # kg
height = 1.65  # m
g = 9.81  # m/s^2
segment_params = {
    'thigh': {'mass': 0.1 * body_mass, 'length': 0.245 * height, 'com_ratio': 0.433, 'I': 0.1 * body_mass * (0.245 * height)**2},
    'shank': {'mass': 0.0465 * body_mass, 'length': 0.246 * height, 'com_ratio': 0.433, 'I': 0.0465 * body_mass * (0.246 * height)**2},
    'foot': {'mass': 0.0145 * body_mass, 'length': 0.152 * height, 'com_ratio': 0.5, 'I': 0.0145 * body_mass * (0.152 * height)**2}
}

# Butterworth filter function
def butter_lowpass_filter(data, cutoff=6, fs=100, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Function to extract gait cycles based on right heel height (RHEE_Z)
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
                print(f"Warning: Failed to interpolate {col} in cycle (possible NaN or insufficient data)")
                norm_data[col] = np.zeros(n_points)
    return pd.DataFrame(norm_data)

# Load and process all CSV files
file_prefix = 'S13_D1_shoe_normal_rd1_'
file_numbers = range(16, 21)
dfs = []

for num in file_numbers:
    filename = f'{file_prefix}{num}.csv'
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        # Convert relevant columns to numeric, assuming positions might be in mm
        for col in df.columns:
            if col != 'time':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # Convert position columns from mm to m (common in motion capture systems)
        position_cols = [col for col in df.columns if any(p in col for p in ['RTOE_', 'RHEE_', 'RKNE_', 'RANK_', 'CoP_'])]
        for col in position_cols:
            df[col] = df[col] / 1000.0  # Convert mm to m
        dfs.append(df)
    else:
        print(f'File {filename} not found.')

# Find the shortest time vector to align all data
min_length = min(len(df['time']) for df in dfs)
time_ref = dfs[np.argmin([len(df['time']) for df in dfs])]['time'].values[:min_length]

# Define required columns
joint_angles = ['RKneeAngles_X', 'RHipAngles_X', 'RAnkleAngles_X']
segment_positions = ['RTOE_X', 'RTOE_Y', 'RTOE_Z', 'RHEE_X', 'RHEE_Y', 'RHEE_Z',
                    'RKNE_X', 'RKNE_Y', 'RKNE_Z', 'RANK_X', 'RANK_Y', 'RANK_Z']
grf_columns = ['FP1_6965 - Force_Fx', 'FP1_6965 - Force_Fy', 'FP1_6965 - Force_Fz',
               'FP2_6966 - Force_Fx', 'FP2_6966 - Force_Fy', 'FP2_6966 - Force_Fz']
cop_columns = ['FP1_6965 - CoP_Cx', 'FP1_6965 - CoP_Cy', 'FP2_6966 - CoP_Cx', 'FP2_6966 - CoP_Cy']
required_columns = joint_angles + segment_positions + grf_columns + cop_columns

# Extract and normalize gait cycles for joint angles
all_cycles = []
numeric_columns = ['time'] + [col for col in dfs[0].columns if col in required_columns and col != 'time']
print("Numeric columns:", numeric_columns)

for df in dfs:
    if 'RHEE_Z' in df.columns:
        cycles = extract_gait_cycles(df, 'time', 'RHEE_Z', min_time_gap=0.5)
        for cycle in cycles:
            norm_cycle = normalize_cycle(cycle, numeric_columns)
            if not norm_cycle.empty and len(norm_cycle) == 101:
                all_cycles.append(norm_cycle)
        print(f"Extracted {len(cycles)} cycles from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")

# Average cycles for joint angles
if all_cycles:
    mean_cycle = pd.concat(all_cycles).groupby('percentage').mean().reset_index()
    print("Columns in mean_cycle:", mean_cycle.columns.tolist())
    print(f"Averaged {len(all_cycles)} cycles")
else:
    print("No cycles detected for joint angles.")
    mean_cycle = pd.DataFrame()

# Apply Butterworth filter to cycle-based joint angles
fs = 100  # Assume 100 Hz for cycle-based data
if not mean_cycle.empty:
    for col in joint_angles:
        if col in mean_cycle.columns and mean_cycle[col].notna().any() and mean_cycle[col].abs().sum() > 0:
            mean_cycle[col] = butter_lowpass_filter(mean_cycle[col], cutoff=6, fs=fs)
        else:
            print(f"Skipping filter for {col} in mean_cycle: contains only NaN or zero values")

# Interpolate only numeric columns
interpolated_dfs = []
for df in dfs:
    interp_data = {'time': time_ref}
    for col in numeric_columns:
        if col != 'time':
            interp_func = interp1d(df['time'], df[col], kind='linear', fill_value="extrapolate")
            interp_data[col] = interp_func(time_ref)
    interp_df = pd.DataFrame(interp_data)
    interpolated_dfs.append(interp_df)

# Compute mean across trials
mean_data = {col: np.mean([df[col].values for df in interpolated_dfs], axis=0) for col in numeric_columns if col != 'time'}
mean_df = pd.DataFrame({'time': time_ref, **mean_data})
print("Columns in mean_df:", mean_df.columns.tolist())

# Check for NaN or infinite values
print("NaN values in mean_df:\n", mean_df.isna().sum())
print("Infinite values in mean_df:\n", np.isinf(mean_df[numeric_columns]).sum())
mean_df.fillna(0, inplace=True)  # Replace NaN with 0
mean_df[numeric_columns] = mean_df[numeric_columns].clip(lower=-1e6, upper=1e6)  # Clip infinite values

# Apply Butterworth filter to kinematic and kinetic data
fs = 1 / np.mean(np.diff(time_ref))
for col in numeric_columns:
    if col != 'time' and mean_df[col].notna().any() and mean_df[col].abs().sum() > 0:
        mean_df[col] = butter_lowpass_filter(mean_df[col], cutoff=6, fs=fs)
    else:
        print(f"Skipping filter for {col}: contains only NaN or zero values")

# 1. Joint Angles Plot
if not mean_cycle.empty:
    print("Joint Angles Plot - Available columns:", [col for col in joint_angles if col in mean_cycle.columns])
    print("Joint Angles Data Range:", {col: (mean_cycle[col].min(), mean_cycle[col].max()) for col in joint_angles if col in mean_cycle.columns})
    plt.figure(figsize=(12, 6))
    for angle in joint_angles:
        if angle in mean_cycle.columns:
            plt.plot(mean_cycle['percentage'], mean_cycle[angle], label=angle)
        else:
            print(f"{angle} not found in mean_cycle")
    plt.xlabel('Gait Cycle (%)')
    plt.ylabel('Angle (degrees)')
    plt.title('Mean Joint Angles Over Gait Cycle')
    plt.legend()
    plt.grid()
    plt.show()
else:
    print("No cycle data for joint angles plot. Falling back to time-based.")
    print("Joint Angles Plot - Available columns:", [col for col in joint_angles if col in mean_df.columns])
    print("Joint Angles Data Range:", {col: (mean_df[col].min(), mean_df[col].max()) for col in joint_angles if col in mean_df.columns})
    plt.figure(figsize=(12, 6))
    for angle in joint_angles:
        if angle in mean_df.columns:
            plt.plot(mean_df['time'], mean_df[angle], label=angle)
        else:
            print(f"{angle} not found in mean_df")
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (degrees)')
    plt.title('Mean Joint Angles Over Time')
    plt.legend()
    plt.grid()
    plt.show()

# 2. Temporal Distribution of GRF
print("GRF Plot - Available columns:", [col for col in grf_columns if col in mean_df.columns])
print("GRF Data Range:", {col: (mean_df[col].min(), mean_df[col].max()) for col in grf_columns if col in mean_df.columns})
plt.figure(figsize=(12, 6))
for col in grf_columns:
    if col in mean_df.columns:
        plt.plot(mean_df['time'], mean_df[col], label=col)
    else:
        print(f"{col} not found in mean_df")
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.title('Mean Ground Reaction Forces (3D Components)')
plt.legend()
plt.grid()
plt.show()

# 3. Inverse Dynamics: 3D Joint Torques
# Convert angles to radians
for angle in joint_angles:
    if angle in mean_df.columns:
        mean_df[f'{angle}_rad'] = np.radians(mean_df[angle])
    else:
        print(f"{angle} not found for radian conversion")

# Calculate angular velocities and accelerations
dt = np.mean(np.diff(mean_df['time']))
for angle in [f'{col}_rad' for col in joint_angles if col in mean_df.columns]:
    if angle in mean_df.columns:
        mean_df[f'{angle}_vel'] = butter_lowpass_filter(np.gradient(mean_df[angle], dt), cutoff=6, fs=fs)
        mean_df[f'{angle}_acc'] = butter_lowpass_filter(np.gradient(mean_df[f'{angle}_vel'], dt), cutoff=6, fs=fs)

# Initialize torque arrays
mean_df['ankle_torque_x'] = 0.0
mean_df['ankle_torque_y'] = 0.0
mean_df['ankle_torque_z'] = 0.0
mean_df['knee_torque_x'] = 0.0
mean_df['knee_torque_y'] = 0.0
mean_df['knee_torque_z'] = 0.0
mean_df['hip_torque_x'] = 0.0
mean_df['hip_torque_y'] = 0.0
mean_df['hip_torque_z'] = 0.0

# 3D Inverse Dynamics
threshold = 10  # N
for i in range(1, len(mean_df)):
    grf = np.array([0, 0, 0])
    cop = np.array([0, 0, 0])
    ankle_pos = np.array([0, 0, 0])
    knee_pos = np.array([0, 0, 0])
    
    # Combine GRF
    if all(col in mean_df.columns for col in ['FP1_6965 - Force_Fx', 'FP1_6965 - Force_Fy', 'FP1_6965 - Force_Fz', 'FP2_6966 - Force_Fx', 'FP2_6966 - Force_Fy', 'FP2_6966 - Force_Fz']):
        grf = np.array([
            mean_df['FP1_6965 - Force_Fx'].iloc[i] + mean_df['FP2_6966 - Force_Fx'].iloc[i],
            mean_df['FP1_6965 - Force_Fy'].iloc[i] + mean_df['FP2_6966 - Force_Fy'].iloc[i],
            mean_df['FP1_6965 - Force_Fz'].iloc[i] + mean_df['FP2_6966 - Force_Fz'].iloc[i]
        ])
    
    # Combine CoP (already in meters from CSV loading)
    if all(col in mean_df.columns for col in ['FP1_6965 - CoP_Cx', 'FP1_6965 - CoP_Cy']) and mean_df['FP1_6965 - Force_Fz'].iloc[i] > threshold:
        cop = np.array([mean_df['FP1_6965 - CoP_Cx'].iloc[i], mean_df['FP1_6965 - CoP_Cy'].iloc[i], 0])

    # Foot segment (ankle torque)
    foot_mass = segment_params['foot']['mass']
    foot_I = segment_params['foot']['I']
    if all(col in mean_df.columns for col in ['RANK_X', 'RANK_Y', 'RANK_Z']):
        ankle_pos = np.array([mean_df['RANK_X'].iloc[i], mean_df['RANK_Y'].iloc[i], mean_df['RANK_Z'].iloc[i]])
    foot_alpha = np.array([mean_df['RAnkleAngles_X_rad_acc'].iloc[i] if 'RAnkleAngles_X_rad_acc' in mean_df.columns else 0, 0, 0])
    r_cop_ankle = cop - ankle_pos
    ankle_torque = np.cross(r_cop_ankle, grf) + foot_I * foot_alpha
    mean_df.loc[i, ['ankle_torque_x', 'ankle_torque_y', 'ankle_torque_z']] = ankle_torque

    # Shank segment (knee torque)
    shank_mass = segment_params['shank']['mass']
    shank_I = segment_params['shank']['I']
    if all(col in mean_df.columns for col in ['RKNE_X', 'RKNE_Y', 'RKNE_Z']):
        knee_pos = np.array([mean_df['RKNE_X'].iloc[i], mean_df['RKNE_Y'].iloc[i], mean_df['RKNE_Z'].iloc[i]])
    shank_alpha = np.array([mean_df['RKneeAngles_X_rad_acc'].iloc[i] if 'RKneeAngles_X_rad_acc' in mean_df.columns else 0, 0, 0])
    r_ankle_knee = ankle_pos - knee_pos
    ankle_reaction_force = grf  # Simplified, ignoring segment weight
    knee_torque = np.cross(r_ankle_knee, ankle_reaction_force) + shank_I * shank_alpha - ankle_torque
    mean_df.loc[i, ['knee_torque_x', 'knee_torque_y', 'knee_torque_z']] = knee_torque

    # Thigh segment (hip torque)
    thigh_mass = segment_params['thigh']['mass']
    thigh_I = segment_params['thigh']['I']
    thigh_alpha = np.array([mean_df['RHipAngles_X_rad_acc'].iloc[i] if 'RHipAngles_X_rad_acc' in mean_df.columns else 0, 0, 0])
    knee_reaction_force = ankle_reaction_force  # Simplified
    hip_torque = thigh_I * thigh_alpha - knee_torque
    mean_df.loc[i, ['hip_torque_x', 'hip_torque_y', 'hip_torque_z']] = hip_torque

# Normalize torques by body weight (Nm/kg)
weight = body_mass * g
mean_df['ankle_torque_x_norm'] = mean_df['ankle_torque_x'] / weight
mean_df['knee_torque_x_norm'] = mean_df['knee_torque_x'] / weight
mean_df['hip_torque_x_norm'] = mean_df['hip_torque_x'] / weight

# Inicializar arrays para fuerzas articulares
mean_df['ankle_reaction_fx'] = 0.0
mean_df['ankle_reaction_fy'] = 0.0
mean_df['ankle_reaction_fz'] = 0.0
mean_df['knee_reaction_fx'] = 0.0
mean_df['knee_reaction_fy'] = 0.0
mean_df['knee_reaction_fz'] = 0.0
mean_df['hip_reaction_fx'] = 0.0
mean_df['hip_reaction_fy'] = 0.0
mean_df['hip_reaction_fz'] = 0.0

# Define critical columns for reaction force calculations
position_cols = ['RANK_X', 'RANK_Y', 'RANK_Z', 'RKNE_X', 'RKNE_Y', 'RKNE_Z']
grf_cols = ['FP1_6965 - Force_Fx', 'FP1_6965 - Force_Fy', 'FP1_6965 - Force_Fz',
            'FP2_6966 - Force_Fx', 'FP2_6966 - Force_Fy', 'FP2_6966 - Force_Fz']
required_cols = position_cols + grf_cols

# Joint reaction force calculation
threshold = 10  # N, to ensure valid GRF
skipped_frames = 0
for i in range(2, len(mean_df)):
    try:
        # Calculate time step and ensure it's valid
        dt = mean_df['time'].iloc[i] - mean_df['time'].iloc[i-1]
        if dt <= 0 or np.isnan(dt):
            print(f"Invalid dt at index {i}: {dt}. Skipping frame.")
            skipped_frames += 1
            continue

        # Check for NaN in required columns for this frame
        frame_data = mean_df[required_cols].iloc[i]
        if frame_data.isna().any():
            print(f"NaN detected in required columns at index {i}: {frame_data.index[frame_data.isna()].tolist()}. Skipping frame.")
            skipped_frames += 1
            continue

        # === Foot Segment ===
        ankle_acc = np.array([
            (mean_df['RANK_X'].iloc[i] - 2 * mean_df['RANK_X'].iloc[i-1] + mean_df['RANK_X'].iloc[i-2]) / dt**2,
            (mean_df['RANK_Y'].iloc[i] - 2 * mean_df['RANK_Y'].iloc[i-1] + mean_df['RANK_Y'].iloc[i-2]) / dt**2,
            (mean_df['RANK_Z'].iloc[i] - 2 * mean_df['RANK_Z'].iloc[i-1] + mean_df['RANK_Z'].iloc[i-2]) / dt**2
        ])
        # Clip accelerations to prevent numerical instability
        ankle_acc = np.clip(ankle_acc, -1e3, 1e3)

        foot_mass = segment_params['foot']['mass']
        foot_weight = np.array([0, 0, -foot_mass * g])
        grf = np.array([
            mean_df['FP1_6965 - Force_Fx'].iloc[i] + mean_df['FP2_6966 - Force_Fx'].iloc[i],
            mean_df['FP1_6965 - Force_Fy'].iloc[i] + mean_df['FP2_6966 - Force_Fy'].iloc[i],
            mean_df['FP1_6965 - Force_Fz'].iloc[i] + mean_df['FP2_6966 - Force_Fz'].iloc[i]
        ])

        # Check if GRF is valid (above threshold)
        if mean_df['FP1_6965 - Force_Fz'].iloc[i] + mean_df['FP2_6966 - Force_Fz'].iloc[i] < threshold:
            #print(f"GRF below threshold at index {i}. Skipping frame.")
            skipped_frames += 1
            continue

        # Compute ankle reaction force: R = GRF - W - m*a
        ankle_reaction_force = grf - foot_weight - foot_mass * ankle_acc
        ankle_reaction_force = np.clip(ankle_reaction_force, -1e6, 1e6)  # Prevent inf
        mean_df.loc[i, ['ankle_reaction_fx', 'ankle_reaction_fy', 'ankle_reaction_fz']] = ankle_reaction_force

        # === Shank Segment ===
        knee_acc = np.array([
            (mean_df['RKNE_X'].iloc[i] - 2 * mean_df['RKNE_X'].iloc[i-1] + mean_df['RKNE_X'].iloc[i-2]) / dt**2,
            (mean_df['RKNE_Y'].iloc[i] - 2 * mean_df['RKNE_Y'].iloc[i-1] + mean_df['RKNE_Y'].iloc[i-2]) / dt**2,
            (mean_df['RKNE_Z'].iloc[i] - 2 * mean_df['RKNE_Z'].iloc[i-1] + mean_df['RKNE_Z'].iloc[i-2]) / dt**2
        ])
        knee_acc = np.clip(knee_acc, -1e3, 1e3)

        shank_mass = segment_params['shank']['mass']
        shank_weight = np.array([0, 0, -shank_mass * g])
        knee_reaction_force = ankle_reaction_force - shank_weight - shank_mass * knee_acc
        knee_reaction_force = np.clip(knee_reaction_force, -1e6, 1e6)
        mean_df.loc[i, ['knee_reaction_fx', 'knee_reaction_fy', 'knee_reaction_fz']] = knee_reaction_force

        # === Thigh Segment ===
        thigh_mass = segment_params['thigh']['mass']
        thigh_weight = np.array([0, 0, -thigh_mass * g])
        hip_reaction_force = knee_reaction_force - thigh_weight  # Simplified, no pelvis acceleration
        hip_reaction_force = np.clip(hip_reaction_force, -1e6, 1e6)
        mean_df.loc[i, ['hip_reaction_fx', 'hip_reaction_fy', 'hip_reaction_fz']] = hip_reaction_force

    except Exception as e:
        print(f"Error at index {i}: {e}. Skipping frame.")
        skipped_frames += 1
        continue

print(f"Skipped {skipped_frames} frames due to NaN, invalid dt, or low GRF.")

# Apply Butterworth filter to reaction forces
for col in ['ankle_reaction_fx', 'ankle_reaction_fy', 'ankle_reaction_fz',
            'knee_reaction_fx', 'knee_reaction_fy', 'knee_reaction_fz',
            'hip_reaction_fx', 'hip_reaction_fy', 'hip_reaction_fz']:
    if mean_df[col].notna().any() and mean_df[col].abs().sum() > 0:
        try:
            mean_df[col] = butter_lowpass_filter(mean_df[col], cutoff=6, fs=fs)
        except Exception as e:
            print(f"Error filtering {col}: {e}. Skipping filter.")
    else:
        print(f"Skipping filter for {col}: contains only NaN or zero values")

# Check for NaN or inf in reaction forces
print("NaN in reaction forces:", mean_df[['ankle_reaction_fz', 'knee_reaction_fz', 'hip_reaction_fz']].isna().sum())
print("Inf in reaction forces:", np.isinf(mean_df[['ankle_reaction_fz', 'knee_reaction_fz', 'hip_reaction_fz']]).sum())

# 4. Numerical Analysis of Torques
torque_cols = ['ankle_torque_x', 'knee_torque_x', 'hip_torque_x']
torque_cols_norm = ['ankle_torque_x_norm', 'knee_torque_x_norm', 'hip_torque_x_norm']
print("\nTorque Statistics (Nm):")
for col in torque_cols:
    print(f"{col}:")
    print(f"  Min: {mean_df[col].min():.2f} Nm")
    print(f"  Max: {mean_df[col].max():.2f} Nm")
    print(f"  Mean: {mean_df[col].mean():.2f} Nm")
    print(f"  Std: {mean_df[col].std():.2f} Nm")

print("\nNormalized Torque Statistics (Nm/kg):")
for col in torque_cols_norm:
    print(f"{col}:")
    print(f"  Min: {mean_df[col].min():.2f} Nm/kg")
    print(f"  Max: {mean_df[col].max():.2f} Nm/kg")
    print(f"  Mean: {mean_df[col].mean():.2f} Nm/kg")
    print(f"  Std: {mean_df[col].std():.2f} Nm/kg")

# Validate torque ranges
expected_ranges = {
    'ankle_torque_x_norm': (0.5, 2.0),  # Nm/kg
    'knee_torque_x_norm': (0.7, 3.0),
    'hip_torque_x_norm': (0.7, 3.5)
}

print("\nTorque Validation:")
for col, (min_exp, max_exp) in expected_ranges.items():
    max_val = mean_df[col].abs().max()
    if max_val < min_exp or max_val > max_exp:
        print(f"Warning: {col} has unrealistic values (max abs: {max_val:.2f} Nm/kg, expected {min_exp}–{max_exp} Nm/kg)")
    else:
        print(f"{col} is within expected range (max abs: {max_val:.2f} Nm/kg)")

# 5. Plot Torques (Sagittal Plane)
print("\nTorque Plot - Data Range (Nm):", {
    'ankle_torque_x': (mean_df['ankle_torque_x'].min(), mean_df['ankle_torque_x'].max()),
    'knee_torque_x': (mean_df['knee_torque_x'].min(), mean_df['knee_torque_x'].max()),
    'hip_torque_x': (mean_df['hip_torque_x'].min(), mean_df['hip_torque_x'].max())
})
print("Torque Plot - Data Range (Nm/kg):", {
    'ankle_torque_x_norm': (mean_df['ankle_torque_x_norm'].min(), mean_df['ankle_torque_x_norm'].max()),
    'knee_torque_x_norm': (mean_df['knee_torque_x_norm'].min(), mean_df['knee_torque_x_norm'].max()),
    'hip_torque_x_norm': (mean_df['hip_torque_x_norm'].min(), mean_df['hip_torque_x_norm'].max())
})
plt.figure(figsize=(12, 6))
plt.plot(mean_df['time'], mean_df['ankle_torque_x'], label='Ankle Torque (X)')
plt.plot(mean_df['time'], mean_df['knee_torque_x'], label='Knee Torque (X)')
plt.plot(mean_df['time'], mean_df['hip_torque_x'], label='Hip Torque (X)')
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.title('Mean Net Joint Torques (Sagittal Plane)')
plt.legend()
plt.grid()
plt.show()

# Plot Normalized Torques
plt.figure(figsize=(12, 6))
plt.plot(mean_df['time'], mean_df['ankle_torque_x_norm'], label='Ankle Torque (X, Normalized)')
plt.plot(mean_df['time'], mean_df['knee_torque_x_norm'], label='Knee Torque (X, Normalized)')
plt.plot(mean_df['time'], mean_df['hip_torque_x_norm'], label='Hip Torque (X, Normalized)')
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm/kg)')
plt.title('Mean Net Joint Torques (Sagittal Plane, Normalized)')
plt.legend()
plt.grid()
plt.show()

# Graficar fuerzas articulares (componente vertical - Z)
plt.figure(figsize=(12, 6))
plt.plot(mean_df['time'], mean_df['ankle_reaction_fz'], label='Ankle Reaction Fz')
plt.plot(mean_df['time'], mean_df['knee_reaction_fz'], label='Knee Reaction Fz')
plt.plot(mean_df['time'], mean_df['hip_reaction_fz'], label='Hip Reaction Fz')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.title('Vertical Joint Reaction Forces')
plt.legend()
plt.grid()
plt.show()

# Imprimir estadísticas
print("\nJoint Reaction Force Statistics (Fz):")
for joint in ['ankle', 'knee', 'hip']:
    valid_data = mean_df[f'{joint}_reaction_fz'][mean_df[f'{joint}_reaction_fz'].notna()]
    if len(valid_data) > 0:
        max_fz = valid_data.max()
        min_fz = valid_data.min()
        mean_fz = valid_data.mean()
        std_fz = valid_data.std()
        print(f"{joint.capitalize()} Fz - Max: {max_fz:.2f} N, Min: {min_fz:.2f} N, Mean: {mean_fz:.2f} N, Std: {std_fz:.2f} N")
    else:
        print(f"{joint.capitalize()} Fz - No valid data (all NaN or zero)")