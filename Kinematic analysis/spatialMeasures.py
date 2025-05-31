import pandas as pd
import numpy as np
from scipy.signal import savgol_filter  # For smoothing the data

# List of file numbers
file_nums = range(16, 21)
files = [f'S13_D1_shoe_normal_rd1_{num}.csv' for num in file_nums]

# Initialize lists for spatial measures
step_lengths = []
right_stride_lengths = []
left_stride_lengths = []
right_max_elevations = []
left_max_elevations = []
base_of_support_widths = []  # New list for base of support width

# Minimum time gap between consecutive heel strikes (in seconds)
min_time_gap = 0.5

for file in files:
    print(f"\nProcessing file: {file}")
    
    # Load the CSV file
    df = pd.read_csv(file)
    
    # Extract relevant columns
    time = df['time'].values
    rhee_z = df['RHEE_Z'].values  # Right heel height
    lhee_z = df['LHEE_Z'].values  # Left heel height
    rhee_x = df['RHEE_X'].values  # Right heel X-coordinate (for base of support width)
    lhee_x = df['LHEE_X'].values  # Left heel X-coordinate (for base of support width)
    rhee_y = df['RHEE_Y'].values  # Right heel Y-coordinate (walking direction)
    lhee_y = df['LHEE_Y'].values  # Left heel Y-coordinate (walking direction)
    
    # Smooth the height data to reduce noise
    rhee_z_smooth = savgol_filter(rhee_z, window_length=11, polyorder=2)
    lhee_z_smooth = savgol_filter(lhee_z, window_length=11, polyorder=2)
    
    # Debug: Check min and max for relevant coordinates
    print(f"Right heel (RHEE_Z) min: {np.min(rhee_z_smooth):.2f}, max: {np.max(rhee_z_smooth):.2f}")
    print(f"Left heel (LHEE_Z) min: {np.min(lhee_z_smooth):.2f}, max: {np.max(lhee_z_smooth):.2f}")
    print(f"Right heel (RHEE_X) min: {np.min(rhee_x):.2f}, max: {np.max(rhee_x):.2f}")
    print(f"Left heel (LHEE_X) min: {np.min(lhee_x):.2f}, max: {np.max(lhee_x):.2f}")
    print(f"Right heel (RHEE_Y) min: {np.min(rhee_y):.2f}, max: {np.max(rhee_y):.2f}")
    print(f"Left heel (LHEE_Y) min: {np.min(lhee_y):.2f}, max: {np.max(lhee_y):.2f}")
    
    # Detect heel strikes (local minima in heel height)
    threshold_right_heel = np.min(rhee_z_smooth) + 0.2 * (np.max(rhee_z_smooth) - np.min(rhee_z_smooth))
    threshold_left_heel = np.min(lhee_z_smooth) + 0.2 * (np.max(lhee_z_smooth) - np.min(lhee_z_smooth))
    
    # Right foot heel strikes
    right_strikes = []
    for i in range(1, len(rhee_z_smooth) - 1):
        if (rhee_z_smooth[i] < threshold_right_heel) and (rhee_z_smooth[i-1] > rhee_z_smooth[i] < rhee_z_smooth[i+1]):
            if not right_strikes or (time[i] - time[right_strikes[-1]]) >= min_time_gap:
                right_strikes.append(i)
    
    # Left foot heel strikes
    left_strikes = []
    for i in range(1, len(lhee_z_smooth) - 1):
        if (lhee_z_smooth[i] < threshold_left_heel) and (lhee_z_smooth[i-1] > lhee_z_smooth[i] < lhee_z_smooth[i+1]):
            if not left_strikes or (time[i] - time[left_strikes[-1]]) >= min_time_gap:
                left_strikes.append(i)
    
    # Debug: Print detected heel strikes and corresponding coordinates
    print(f"Right foot strikes: {len(right_strikes)} at times {time[right_strikes] if right_strikes else 'None'}")
    if right_strikes:
        print(f"Right foot RHEE_X at strikes: {[rhee_x[i] for i in right_strikes]}")
        print(f"Right foot RHEE_Y at strikes: {[rhee_y[i] for i in right_strikes]}")
    print(f"Left foot strikes: {len(left_strikes)} at times {time[left_strikes] if left_strikes else 'None'}")
    if left_strikes:
        print(f"Left foot LHEE_X at strikes: {[lhee_x[i] for i in left_strikes]}")
        print(f"Left foot LHEE_Y at strikes: {[lhee_y[i] for i in left_strikes]}")
    
    # Combine strikes for step length and base of support width calculations
    file_strikes = []
    for i in right_strikes:
        file_strikes.append((time[i], 'right', rhee_x[i], rhee_y[i], rhee_z[i]))
    for i in left_strikes:
        file_strikes.append((time[i], 'left', lhee_x[i], lhee_y[i], lhee_z[i]))
    
    file_strikes.sort(key=lambda x: x[0])
    
    print(f"Strikes for {file} (sorted by time):")
    for strike in file_strikes:
        print(f"Time: {strike[0]:.2f}, Foot: {strike[1]}, X: {strike[2]:.2f}, Y: {strike[3]:.2f}")
    
    # Step Length: Using Y-axis
    file_step_lengths = []
    for i in range(len(file_strikes) - 1):
        if file_strikes[i][1] != file_strikes[i+1][1]:  # Opposite feet
            y1 = file_strikes[i][3]
            y2 = file_strikes[i+1][3]
            step_length = abs(y2 - y1)
            file_step_lengths.append(step_length)
    
    if file_step_lengths:
        step_lengths.extend(file_step_lengths)
        print(f"Step Lengths in {file}: {file_step_lengths}")
    
    # Base of Support Width: Using X-axis during approximate double support phases
    file_base_widths = []
    for i in range(len(file_strikes) - 1):
        if file_strikes[i][1] != file_strikes[i+1][1]:  # Opposite feet (approximating double support)
            x1 = file_strikes[i][2]  # RHEE_X or LHEE_X of first strike
            x2 = file_strikes[i+1][2]  # RHEE_X or LHEE_X of second strike
            base_width = abs(x2 - x1)
            file_base_widths.append(base_width)
    
    if file_base_widths:
        base_of_support_widths.extend(file_base_widths)
        print(f"Base of Support Widths in {file}: {file_base_widths}")
    
    # Stride Length and Maximum Heel Elevation
    file_right_stride_lengths = []
    for i in range(len(right_strikes) - 1):
        stride_length = abs(rhee_y[right_strikes[i+1]] - rhee_y[right_strikes[i]])
        file_right_stride_lengths.append(stride_length)
        # Maximum heel elevation in this cycle
        cycle_rhee_z = rhee_z[right_strikes[i]:right_strikes[i+1]+1]
        max_elevation = np.max(cycle_rhee_z)
        right_max_elevations.append(max_elevation)
    
    if file_right_stride_lengths:
        right_stride_lengths.extend(file_right_stride_lengths)
        print(f"Right Stride Lengths in {file}: {file_right_stride_lengths}")
    
    file_left_stride_lengths = []
    for i in range(len(left_strikes) - 1):
        stride_length = abs(lhee_y[left_strikes[i+1]] - lhee_y[left_strikes[i]])
        file_left_stride_lengths.append(stride_length)
        # Maximum heel elevation in this cycle
        cycle_lhee_z = lhee_z[left_strikes[i]:left_strikes[i+1]+1]
        max_elevation = np.max(cycle_lhee_z)
        left_max_elevations.append(max_elevation)
    
    if file_left_stride_lengths:
        left_stride_lengths.extend(file_left_stride_lengths)
        print(f"Left Stride Lengths in {file}: {file_left_stride_lengths}")

# --- Final Results ---
mean_step_length = np.mean(step_lengths) if step_lengths else 0
mean_right_stride_length = np.mean(right_stride_lengths) if right_stride_lengths else 0
mean_left_stride_length = np.mean(left_stride_lengths) if left_stride_lengths else 0
mean_right_max_elevation = np.mean(right_max_elevations) if right_max_elevations else 0
mean_left_max_elevation = np.mean(left_max_elevations) if left_max_elevations else 0
mean_base_of_support_width = np.mean(base_of_support_widths) if base_of_support_widths else 0

# --- Symmetry Calculations ---
def symmetry_index(right, left):
    return (abs(right - left) / ((right + left) / 2)) * 100 if (right + left) != 0 else 0

stride_symmetry = symmetry_index(mean_right_stride_length, mean_left_stride_length)
elevation_symmetry = symmetry_index(mean_right_max_elevation, mean_left_max_elevation)

# --- Display Results ---
print(f"\nFinal Results:")
print(f"Mean Step Length: {mean_step_length:.2f} mm")
print(f"Mean Right Stride Length: {mean_right_stride_length:.2f} mm")
print(f"Mean Left Stride Length: {mean_left_stride_length:.2f} mm")
print(f"Stride Length Symmetry Index: {stride_symmetry:.2f} %")
print(f"Mean Right Foot Maximum Heel Elevation: {mean_right_max_elevation:.2f} mm")
print(f"Mean Left Foot Maximum Heel Elevation: {mean_left_max_elevation:.2f} mm")
print(f"Max Heel Elevation Symmetry Index: {elevation_symmetry:.2f} %")
print(f"Mean Base of Support Width: {mean_base_of_support_width:.2f} mm")
