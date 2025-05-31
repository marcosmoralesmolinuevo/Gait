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
bw = body_mass * g  # Body weight (N)
bwh = body_mass * g * height  # For moment normalization

# Dempster's data
segment_params = {
    'thigh': {'mass_ratio': 0.100, 'length_ratio': 0.245, 'com_ratio': 0.433, 'radius_gyration_ratio': 0.323},
    'shank': {'mass_ratio': 0.0465, 'length_ratio': 0.246, 'com_ratio': 0.433, 'radius_gyration_ratio': 0.302},
    'foot': {'mass_ratio': 0.0145, 'length_ratio': 0.152, 'com_ratio': 0.500, 'radius_gyration_ratio': 0.475}
}

# Butterworth filter
def butter_lowpass_filter(data, cutoff=10, fs=100, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Extract gait cycles
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

# Normalize cycle
def normalize_cycle(cycle, columns, n_points=101):
    time = cycle['time'].values
    cycle_percentage = np.linspace(0, 100, n_points)
    norm_data = {'percentage': cycle_percentage}
    for col in columns:
        if col in cycle.columns and col != 'time':
            try:
                data = cycle[col].values
                if np.all(np.isnan(data)):
                    norm_data[col] = np.zeros(n_points)
                else:
                    interp_func = interp1d(np.linspace(0, 100, len(time)), data, kind='linear', fill_value="extrapolate")
                    norm_data[col] = interp_func(cycle_percentage)
            except Exception as e:
                print(f"Error normalizing {col}: {e}")
                norm_data[col] = np.zeros(n_points)
    return pd.DataFrame(norm_data)

# Select force plate cycle
def select_force_plate_cycle(cycles, grf_cols, filename):
    selected_cycle = None
    max_grf = 0
    for cycle in cycles:
        if all(col in cycle.columns for col in ['FP1_6965 - Force_Fz', 'FP2_6966 - Force_Fz']):
            total_fz = cycle['FP1_6965 - Force_Fz'] + cycle['FP2_6966 - Force_Fz']
            if len(total_fz[total_fz > 20]) > 10:  # Threshold for cycle selection only
                peak_grf = total_fz.max()
                if peak_grf > max_grf:
                    max_grf = peak_grf
                    selected_cycle = cycle
    if selected_cycle is not None:
        print(f"Selected cycle from {filename}, Peak GRF: {max_grf:.2f} N")
    return selected_cycle

# Load CSVs
file_prefix = 'S13_D1_shoe_normal_rd1_'
file_numbers = range(16, 21)
dfs = []
position_cols = ['RTOE_X', 'RTOE_Y', 'RTOE_Z', 'RHEE_X', 'RHEE_Y', 'RHEE_Z', 'RKNE_X', 'RKNE_Y', 'RKNE_Z', 'RANK_X', 'RANK_Y', 'RANK_Z']
grf_cols = ['FP1_6965 - Force_Fx', 'FP1_6965 - Force_Fz', 'FP1_6965 - CoP_Cx', 'FP2_6966 - Force_Fx', 'FP2_6966 - Force_Fz', 'FP2_6966 - CoP_Cx']
angle_cols = ['RAnkleAngles_X', 'RKneeAngles_X', 'RHipAngles_X']
angular_vel_cols = ['RAnkleAngles_X_v', 'RKneeAngles_X_v', 'RHipAngles_X_v']
numeric_columns = ['time'] + position_cols + grf_cols + angle_cols + angular_vel_cols

for num in file_numbers:
    filename = f'{file_prefix}{num}.csv'
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        # Convert positions and CoP to meters
        for col in position_cols + ['FP1_6965 - CoP_Cx', 'FP2_6966 - CoP_Cx']:
            if col in df.columns:
                df[col] = df[col] / 1000.0
        # Process angular velocities
        for col in angular_vel_cols:
            if col in df.columns:
                print(f"Raw {col}: Max={df[col].max():.2f}, Min={df[col].min():.2f} deg/s")
                df[col] = np.radians(df[col])  # Degrees/s to rad/s
                df[col] = df[col].interpolate(method='linear', limit_direction='both')  # No fillna(0)
                df[col] = butter_lowpass_filter(df[col], cutoff=10, fs=100, order=2)
                df[col] = np.clip(df[col], -10, 10)  # ±573 deg/s
                print(f"Processed {col}: Max={df[col].max():.2f}, Min={df[col].min():.2f} rad/s")
        # Interpolate other columns
        available_cols = [col for col in numeric_columns if col in df.columns and col not in angular_vel_cols]
        df[available_cols] = df[available_cols].interpolate(method='linear', axis=0, limit_direction='both').fillna(0)
        dfs.append(df)
        print(f"Loaded {filename}")
    else:
        print(f"File {filename} not found.")

# Debug GRF
for df, num in zip(dfs, file_numbers):
    total_fz = df['FP1_6965 - Force_Fz'] + df['FP2_6966 - Force_Fz']
    print(f"{file_prefix}{num}: Max GRF={total_fz.max():.2f} N")

# Extract cycles
selected_cycles = []
for df, num in zip(dfs, file_numbers):
    filename = f'{file_prefix}{num}.csv'
    if 'RHEE_Z' in df.columns:
        cycles = extract_gait_cycles(df, 'time', 'RHEE_Z', min_time_gap=0.5)
        print(f"Extracted {len(cycles)} cycles from {filename}")
        force_cycle = select_force_plate_cycle(cycles, grf_cols, filename)
        if force_cycle is not None:
            norm_cycle = normalize_cycle(force_cycle, numeric_columns)
            if not norm_cycle.empty and len(norm_cycle) == 101:
                # Re-filter angular velocities
                for col in angular_vel_cols:
                    if col in norm_cycle.columns:
                        norm_cycle[col] = butter_lowpass_filter(norm_cycle[col], cutoff=10, fs=100, order=2)
                        norm_cycle[col] = np.clip(norm_cycle[col], -10, 10)
                selected_cycles.append(norm_cycle)
        else:
            print(f"No valid force plate cycle in {filename}")

print(f"Total selected cycles: {len(selected_cycles)}")

# Inertial parameters
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
    radius_gyration = segment_params[segment]['radius_gyration_ratio'] * length
    moment_inertia = mass * radius_gyration**2
    inertial_params[segment] = {'mass': mass, 'length': length, 'com_distance': com_distance, 'moment_inertia': moment_inertia}

# Calculate moments and power
moments = {'ankle': [], 'knee': [], 'hip': []}
powers = {'ankle': [], 'knee': [], 'hip': []}
fs = 100
dt = 1 / fs

for cycle in selected_cycles:
    cycle_moments = {'ankle': np.zeros(101), 'knee': np.zeros(101), 'hip': np.zeros(101)}
    cycle_powers = {'ankle': np.zeros(101), 'knee': np.zeros(101), 'hip': np.zeros(101)}
    last_cop_x = 0  # Track last known CoP position

    for i in range(2, 101):
        required_cols = ['RANK_X', 'RANK_Z', 'RKNE_X', 'RKNE_Z', 'RTOE_X', 'RTOE_Z'] + grf_cols
        if any(col not in cycle.columns or np.isnan(cycle[col].iloc[i]) for col in required_cols):
            continue

        grf = np.array([
            cycle['FP1_6965 - Force_Fx'].iloc[i] + cycle['FP2_6966 - Force_Fx'].iloc[i],
            cycle['FP1_6965 - Force_Fz'].iloc[i] + cycle['FP2_6966 - Force_Fz'].iloc[i]
        ])
        fz1 = cycle['FP1_6965 - Force_Fz'].iloc[i]
        fz2 = cycle['FP2_6966 - Force_Fz'].iloc[i]
        total_fz = fz1 + fz2
        if total_fz > 0 and 'FP1_6965 - CoP_Cx' in cycle.columns and 'FP2_6966 - CoP_Cx' in cycle.columns:
            cop_x = (cycle['FP1_6965 - CoP_Cx'].iloc[i] * fz1 + cycle['FP2_6966 - CoP_Cx'].iloc[i] * fz2) / total_fz
            last_cop_x = cop_x
        else:
            # During swing (low GRF), assume zero GRF and use last known CoP
            grf = np.array([0, 0])
            cop_x = last_cop_x
        cop = np.array([cop_x, 0])

        # Foot
        ankle_pos = np.array([cycle['RANK_X'].iloc[i], cycle['RANK_Z'].iloc[i]])
        ankle_acc = np.array([
            (cycle['RANK_X'].iloc[i] - 2*cycle['RANK_X'].iloc[i-1] + cycle['RANK_X'].iloc[i-2]) / dt**2,
            (cycle['RANK_Z'].iloc[i] - 2*cycle['RANK_Z'].iloc[i-1] + cycle['RANK_Z'].iloc[i-2]) / dt**2
        ])
        ankle_acc = np.clip(ankle_acc, -1e3, 1e3)
        ankle_angular_acc = ((cycle['RAnkleAngles_X_v'].iloc[i] - cycle['RAnkleAngles_X_v'].iloc[i-1]) / dt 
                             if 'RAnkleAngles_X_v' in cycle.columns else 0)
        foot_mass = inertial_params['foot']['mass']
        foot_com_dist = inertial_params['foot']['com_distance']
        foot_moi = inertial_params['foot']['moment_inertia']
        foot_weight = np.array([0, -foot_mass * g])
        ankle_reaction = grf - foot_weight - foot_mass * ankle_acc
        r_cop_ankle = cop - ankle_pos
        r_com_ankle = np.array([0, -foot_com_dist])
        ankle_moment = (np.cross(r_cop_ankle, grf) + 
                        np.cross(r_com_ankle, -foot_weight - foot_mass * ankle_acc) - 
                        foot_moi * ankle_angular_acc)
        cycle_moments['ankle'][i] = ankle_moment

        # Shank
        knee_pos = np.array([cycle['RKNE_X'].iloc[i], cycle['RKNE_Z'].iloc[i]])
        knee_acc = np.array([
            (cycle['RKNE_X'].iloc[i] - 2*cycle['RKNE_X'].iloc[i-1] + cycle['RKNE_X'].iloc[i-2]) / dt**2,
            (cycle['RKNE_Z'].iloc[i] - 2*cycle['RKNE_Z'].iloc[i-1] + cycle['RKNE_Z'].iloc[i-2]) / dt**2
        ])
        knee_acc = np.clip(knee_acc, -1e3, 1e3)
        knee_angular_acc = ((cycle['RKneeAngles_X_v'].iloc[i] - cycle['RKneeAngles_X_v'].iloc[i-1]) / dt 
                            if 'RKneeAngles_X_v' in cycle.columns else 0)
        shank_mass = inertial_params['shank']['mass']
        shank_com_dist = inertial_params['shank']['com_distance']
        shank_moi = inertial_params['shank']['moment_inertia']
        shank_weight = np.array([0, -shank_mass * g])
        knee_reaction = ankle_reaction - shank_weight - shank_mass * knee_acc
        r_ankle_knee = ankle_pos - knee_pos
        r_com_knee = np.array([0, -shank_com_dist])
        knee_moment = (ankle_moment + 
                       np.cross(r_ankle_knee, ankle_reaction) + 
                       np.cross(r_com_knee, -shank_weight - shank_mass * knee_acc) - 
                       shank_moi * knee_angular_acc)
        cycle_moments['knee'][i] = knee_moment

        # Thigh
        thigh_mass = inertial_params['thigh']['mass']
        thigh_com_dist = inertial_params['thigh']['com_distance']
        thigh_moi = inertial_params['thigh']['moment_inertia']
        thigh_weight = np.array([0, -thigh_mass * g])
        hip_reaction = knee_reaction - thigh_weight
        hip_angular_acc = ((cycle['RHipAngles_X_v'].iloc[i] - cycle['RHipAngles_X_v'].iloc[i-1]) / dt 
                           if 'RHipAngles_X_v' in cycle.columns else 0)
        r_knee_hip = np.array([0, inertial_params['thigh']['length']])
        r_com_hip = np.array([0, -thigh_com_dist])
        hip_moment = (knee_moment + 
                      np.cross(r_knee_hip, knee_reaction) + 
                      np.cross(r_com_hip, -thigh_weight) - 
                      thigh_moi * hip_angular_acc)
        cycle_moments['hip'][i] = hip_moment

        # Power
        if 'RAnkleAngles_X_v' in cycle.columns and not np.isnan(cycle['RAnkleAngles_X_v'].iloc[i]):
            cycle_powers['ankle'][i] = ankle_moment * cycle['RAnkleAngles_X_v'].iloc[i]
        if 'RKneeAngles_X_v' in cycle.columns and not np.isnan(cycle['RKneeAngles_X_v'].iloc[i]):
            cycle_powers['knee'][i] = knee_moment * cycle['RKneeAngles_X_v'].iloc[i]
        if 'RHipAngles_X_v' in cycle.columns and not np.isnan(cycle['RHipAngles_X_v'].iloc[i]):
            cycle_powers['hip'][i] = hip_moment * cycle['RHipAngles_X_v'].iloc[i]

    # Filter moments and powers
    for key in cycle_moments:
        if np.sum(np.abs(cycle_moments[key])) > 0:
            cycle_moments[key] = butter_lowpass_filter(cycle_moments[key], cutoff=10, fs=fs)
    for key in cycle_powers:
        if np.sum(np.abs(cycle_powers[key])) > 0:
            cycle_powers[key] = butter_lowpass_filter(cycle_powers[key], cutoff=10, fs=fs)

    # Debug zeros
    for key in cycle_moments:
        zero_count = np.sum(cycle_moments[key] == 0)
        print(f"Cycle Moments {key}: Zero count={zero_count}")
        # Log regions where moments are zero
        if zero_count > 0:
            zero_indices = np.where(cycle_moments[key] == 0)[0]
            print(f"  Zero moments at percentages: {zero_indices}")
    for key in cycle_powers:
        zero_count = np.sum(cycle_powers[key] == 0)
        print(f"Cycle Powers {key}: Zero count={zero_count}")
        if zero_count > 0:
            zero_indices = np.where(cycle_powers[key] == 0)[0]
            print(f"  Zero powers at percentages: {zero_indices}")

    for key in moments:
        moments[key].append(cycle_moments[key])
        powers[key].append(cycle_powers[key])

# Average
avg_moments = {key: np.nanmean(moments[key], axis=0) for key in moments}
avg_powers = {key: np.nanmean(powers[key], axis=0) for key in powers}

# Normalize
avg_moments_norm = {key: avg_moments[key] / bwh for key in avg_moments}
avg_powers_norm = {key: avg_powers[key] / bw for key in avg_powers}

# Phase analysis
stance_idx = slice(0, 61)
swing_idx = slice(61, 101)

print("\nJoint Moment Statistics (%BW·h):")
for joint in ['ankle', 'knee', 'hip']:
    valid_data = avg_moments_norm[joint][~np.isnan(avg_moments_norm[joint])]
    stance_data = avg_moments_norm[joint][stance_idx][~np.isnan(avg_moments_norm[joint][stance_idx])]
    swing_data = avg_moments_norm[joint][swing_idx][~np.isnan(avg_moments_norm[joint][swing_idx])]
    if len(valid_data) > 0:
        print(f"{joint.capitalize()}:")
        print(f"  Full Cycle: Max={valid_data.max()*100:.2f}%BW·h, Min={valid_data.min()*100:.2f}%BW·h, Mean={valid_data.mean()*100:.2f}%BW·h, Std={valid_data.std()*100:.2f}%BW·h")
        if len(stance_data) > 0:
            print(f"  Stance: Max={stance_data.max()*100:.2f}%BW·h, Min={stance_data.min()*100:.2f}%BW·h")
        if len(swing_data) > 0:
            print(f"  Swing: Max={swing_data.max()*100:.2f}%BW·h, Min={swing_data.min()*100:.2f}%BW·h")

print("\nJoint Power Statistics (%W/BW):")
for joint in ['ankle', 'knee', 'hip']:
    valid_data = avg_powers_norm[joint][~np.isnan(avg_powers_norm[joint])]
    stance_data = avg_powers_norm[joint][stance_idx][~np.isnan(avg_powers_norm[joint][stance_idx])]
    swing_data = avg_powers_norm[joint][swing_idx][~np.isnan(avg_powers_norm[joint][swing_idx])]
    pos_power = valid_data[valid_data > 0].sum() / len(valid_data) * 100 if len(valid_data[valid_data > 0]) > 0 else 0
    neg_power = valid_data[valid_data < 0].sum() / len(valid_data) * 100 if len(valid_data[valid_data < 0]) > 0 else 0
    if len(valid_data) > 0:
        print(f"{joint.capitalize()}:")
        print(f"  Full Cycle: Max={valid_data.max()*100:.2f}%W/BW, Min={valid_data.min()*100:.2f}%W/BW, Mean={valid_data.mean()*100:.2f}%W/BW, Std={valid_data.std()*100:.2f}%W/BW")
        print(f"  Positive Power: {pos_power:.2f}%W/BW")
        print(f"  Negative Power: {neg_power:.2f}%W/BW")
        if len(stance_data) > 0:
            print(f"  Stance: Max={stance_data.max()*100:.2f}%W/BW, Min={stance_data.min()*100:.2f}%W/BW")
        if len(swing_data) > 0:
            print(f"  Swing: Max={swing_data.max()*100:.2f}%W/BW, Min={swing_data.min()*100:.2f}%W/BW")

# Debug angular velocities
for cycle in selected_cycles:
    for col in angular_vel_cols:
        if col in cycle.columns:
            zero_count = np.sum(cycle[col] == 0)
            print(f"{col}: Max={np.max(cycle[col]):.2f} rad/s, Min={np.min(cycle[col]):.2f} rad/s, Zero count={zero_count}")

# Plot average powers and moments
percentage = np.linspace(0, 100, 101)
plt.figure(figsize=(12, 6))
for joint in avg_moments_norm:
    plt.plot(percentage, avg_moments_norm[joint]*100, label=f'{joint.capitalize()} Moment')
plt.xlabel('Gait Cycle (%)')
plt.ylabel('Moment (%BW·h)')
plt.title('Average Joint Moments')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))
for joint in avg_powers_norm:
    plt.plot(percentage, avg_powers_norm[joint]*100, label=f'{joint.capitalize()} Power')
plt.ylabel('Power (%W/BW)')
plt.title('Average Joint Powers')
plt.legend()
plt.grid()
plt.show()