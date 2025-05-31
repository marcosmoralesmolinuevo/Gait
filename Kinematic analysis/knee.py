import pandas as pd
import numpy as np
from scipy.signal import savgol_filter  # For smoothing the data
from scipy.interpolate import interp1d  # For interpolating angles over gait cycle
import matplotlib.pyplot as plt

# List of file numbers
file_nums = range(16, 21)
files = [f'S13_D1_shoe_normal_rd1_{num}.csv' for num in file_nums]

# Lists to store interpolated angle data for plotting
r_knee_flex_cycles = []  # Right knee flexion/extension cycles
l_knee_flex_cycles = []  # Left knee flexion/extension cycles

# Minimum time gap between consecutive heel strikes (in seconds)
min_time_gap = 0.5

# Grid for gait cycle percentage (0% to 100%)
pct_grid = np.linspace(0, 100, 101)

for file in files:
    print(f"\nProcessing file: {file}")
    
    # Load the CSV file
    df = pd.read_csv(file)
    
    # Extract relevant columns
    time = df['time'].values
    rhee_z = df['RHEE_Z'].values  # Right heel height for gait cycle detection
    lhee_z = df['LHEE_Z'].values  # Left heel height for gait cycle detection
    r_knee_flex = df['RKneeAngles_X'].values  # Right knee flexion/extension
    l_knee_flex = df['LKneeAngles_X'].values  # Left knee flexion/extension
    
    # Smooth the knee angle data to reduce noise
    r_knee_flex_smooth = savgol_filter(r_knee_flex, window_length=11, polyorder=2)
    l_knee_flex_smooth = savgol_filter(l_knee_flex, window_length=11, polyorder=2)
    
    # --- Detect Gait Cycles (using heel strikes) ---
    # Smooth the height data to reduce noise
    rhee_z_smooth = savgol_filter(rhee_z, window_length=11, polyorder=2)
    lhee_z_smooth = savgol_filter(lhee_z, window_length=11, polyorder=2)
    
    # Detect heel strikes (local minima in heel height)
    threshold_right_heel = np.nanmin(rhee_z_smooth) + 0.2 * (np.nanmax(rhee_z_smooth) - np.nanmin(rhee_z_smooth))
    threshold_left_heel = np.nanmin(lhee_z_smooth) + 0.2 * (np.nanmax(lhee_z_smooth) - np.nanmin(lhee_z_smooth))
    
    right_strikes = []
    for i in range(1, len(rhee_z_smooth) - 1):
        if (rhee_z_smooth[i] < threshold_right_heel) and (rhee_z_smooth[i-1] > rhee_z_smooth[i] < rhee_z_smooth[i+1]):
            if not right_strikes or (time[i] - time[right_strikes[-1]]) >= min_time_gap:
                right_strikes.append(i)
    
    left_strikes = []
    for i in range(1, len(lhee_z_smooth) - 1):
        if (lhee_z_smooth[i] < threshold_left_heel) and (lhee_z_smooth[i-1] > lhee_z_smooth[i] < lhee_z_smooth[i+1]):
            if not left_strikes or (time[i] - time[left_strikes[-1]]) >= min_time_gap:
                left_strikes.append(i)
    
    # Debugging: Print the number of detected heel strikes
    print(f"Number of right heel strikes: {len(right_strikes)}")
    print(f"Number of left heel strikes: {len(left_strikes)}")
    
    # Combine strikes to define gait cycles
    file_strikes = []
    for i in right_strikes:
        file_strikes.append((time[i], 'right'))
    for i in left_strikes:
        file_strikes.append((time[i], 'left'))
    
    file_strikes.sort(key=lambda x: x[0])
    
    # --- Interpolate Right Knee Angles Over Gait Cycle ---
    if len(right_strikes) >= 2:
        cycle_starts = [time[i] for i in right_strikes]
        r_knee_flex_file_cycles = []
        
        for i in range(len(cycle_starts) - 1):
            cycle_start = cycle_starts[i]
            cycle_end = cycle_starts[i+1]
            cycle_duration = cycle_end - cycle_start
            
            # Find indices for this cycle
            cycle_indices = (time >= cycle_start) & (time <= cycle_end)
            cycle_time = time[cycle_indices]
            cycle_pct = ((cycle_time - cycle_start) / cycle_duration) * 100
            
            # Interpolate angles over the gait cycle percentage
            if len(cycle_pct) > 1 and np.ptp(cycle_pct) > 0.01:
                interp_r_knee_flex = interp1d(cycle_pct, r_knee_flex_smooth[cycle_indices], bounds_error=False, fill_value="extrapolate")
                r_knee_flex_file_cycles.append(interp_r_knee_flex(pct_grid))
        
        if r_knee_flex_file_cycles:
            r_knee_flex_cycles.append(np.nanmean(r_knee_flex_file_cycles, axis=0))
    else:
        print(f"Warning: Fewer than 2 right heel strikes detected in {file}. Cannot define right gait cycles for plotting.")
    
    # --- Interpolate Left Knee Angles Over Gait Cycle ---
    if len(left_strikes) >= 2:
        cycle_starts = [time[i] for i in left_strikes]
        l_knee_flex_file_cycles = []
        
        for i in range(len(cycle_starts) - 1):
            cycle_start = cycle_starts[i]
            cycle_end = cycle_starts[i+1]
            cycle_duration = cycle_end - cycle_start
            
            # Find indices for this cycle
            cycle_indices = (time >= cycle_start) & (time <= cycle_end)
            cycle_time = time[cycle_indices]
            cycle_pct = ((cycle_time - cycle_start) / cycle_duration) * 100
            
            # Interpolate angles over the gait cycle percentage
            if len(cycle_pct) > 1 and np.ptp(cycle_pct) > 0.01:
                interp_l_knee_flex = interp1d(cycle_pct, l_knee_flex_smooth[cycle_indices], bounds_error=False, fill_value="extrapolate")
                l_knee_flex_file_cycles.append(interp_l_knee_flex(pct_grid))
        
        if l_knee_flex_file_cycles:
            l_knee_flex_cycles.append(np.nanmean(l_knee_flex_file_cycles, axis=0))
    else:
        print(f"Warning: Fewer than 2 left heel strikes detected in {file}. Cannot define left gait cycles for plotting.")

# --- Compute Mean Angles Across Files for Plotting and Metrics ---
mean_r_knee_flex = np.nanmean(r_knee_flex_cycles, axis=0) if r_knee_flex_cycles else np.full(101, np.nan)
mean_l_knee_flex = np.nanmean(l_knee_flex_cycles, axis=0) if l_knee_flex_cycles else np.full(101, np.nan)

# --- Calculate Metrics from the Averaged Curves ---
# Right Knee
mean_r_max_flex = np.nanmax(mean_r_knee_flex) if not np.all(np.isnan(mean_r_knee_flex)) else np.nan
mean_r_min_flex = np.nanmin(mean_r_knee_flex) if not np.all(np.isnan(mean_r_knee_flex)) else np.nan
mean_r_rom_flex = mean_r_max_flex - mean_r_min_flex if not np.isnan(mean_r_max_flex) and not np.isnan(mean_r_min_flex) else np.nan

# Find timing of max and min for right knee
r_max_flex_idx = np.nanargmax(mean_r_knee_flex)
r_min_flex_idx = np.nanargmin(mean_r_knee_flex)
mean_r_max_flex_time = pct_grid[r_max_flex_idx] if not np.isnan(mean_r_knee_flex[r_max_flex_idx]) else np.nan
mean_r_min_flex_time = pct_grid[r_min_flex_idx] if not np.isnan(mean_r_knee_flex[r_min_flex_idx]) else np.nan

# Left Knee
mean_l_max_flex = np.nanmax(mean_l_knee_flex) if not np.all(np.isnan(mean_l_knee_flex)) else np.nan
mean_l_min_flex = np.nanmin(mean_l_knee_flex) if not np.all(np.isnan(mean_l_knee_flex)) else np.nan
mean_l_rom_flex = mean_l_max_flex - mean_l_min_flex if not np.isnan(mean_l_max_flex) and not np.isnan(mean_l_min_flex) else np.nan

# Find timing of max and min for left knee
l_max_flex_idx = np.nanargmax(mean_l_knee_flex)
l_min_flex_idx = np.nanargmin(mean_l_knee_flex)
mean_l_max_flex_time = pct_grid[l_max_flex_idx] if not np.isnan(mean_l_knee_flex[l_max_flex_idx]) else np.nan
mean_l_min_flex_time = pct_grid[l_min_flex_idx] if not np.isnan(mean_l_knee_flex[l_min_flex_idx]) else np.nan

# --- Compensation Between Sides ---
# Differences in ROM
diff_rom_flex = abs(mean_r_rom_flex - mean_l_rom_flex) if not np.isnan(mean_r_rom_flex) and not np.isnan(mean_l_rom_flex) else np.nan
# Differences in peak angles
diff_max_flex = abs(mean_r_max_flex - mean_l_max_flex) if not np.isnan(mean_r_max_flex) and not np.isnan(mean_l_max_flex) else np.nan
diff_min_flex = abs(mean_r_min_flex - mean_l_min_flex) if not np.isnan(mean_r_min_flex) and not np.isnan(mean_l_min_flex) else np.nan

# --- Plot Knee Flexion/Extension Over Gait Cycle ---
plt.figure(figsize=(10, 6))
plt.plot(pct_grid, mean_r_knee_flex, label='Right Knee', color='blue')
plt.plot(pct_grid, mean_l_knee_flex, label='Left Knee', color='red')
plt.title('Knee Flexion/Extension (Sagittal Plane) Over Gait Cycle')
plt.xlabel('Gait Cycle (%)')
plt.ylabel('Angle (degrees)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Final Results ---
print(f"\nFinal Results (Averaged Across Files with Valid Data):")

print(f"\nRight Knee:")
print(f"Flexion/Extension (Sagittal Plane):")
print(f"  Mean Maximum Angle: {mean_r_max_flex:.2f} degrees")
print(f"  Mean Minimum Angle: {mean_r_min_flex:.2f} degrees")
print(f"  Mean Range of Motion: {mean_r_rom_flex:.2f} degrees")
print(f"  Mean Timing of Maximum (Gait Cycle %): {mean_r_max_flex_time:.2f}%")
print(f"  Mean Timing of Minimum (Gait Cycle %): {mean_r_min_flex_time:.2f}%")

print(f"\nLeft Knee:")
print(f"Flexion/Extension (Sagittal Plane):")
print(f"  Mean Maximum Angle: {mean_l_max_flex:.2f} degrees")
print(f"  Mean Minimum Angle: {mean_l_min_flex:.2f} degrees")
print(f"  Mean Range of Motion: {mean_l_rom_flex:.2f} degrees")
print(f"  Mean Timing of Maximum (Gait Cycle %): {mean_l_max_flex_time:.2f}%")
print(f"  Mean Timing of Minimum (Gait Cycle %): {mean_l_min_flex_time:.2f}%")

print(f"\nCompensation Between Sides:")
print(f"Flexion/Extension:")
print(f"  Difference in ROM: {diff_rom_flex:.2f} degrees")
print(f"  Difference in Max Angle: {diff_max_flex:.2f} degrees")
print(f"  Difference in Min Angle: {diff_min_flex:.2f} degrees")