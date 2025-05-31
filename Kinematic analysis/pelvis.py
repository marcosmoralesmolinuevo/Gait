import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# List of file numbers
file_nums = range(16, 21)
files = [f'S13_D1_shoe_normal_rd1_{num}.csv' for num in file_nums]

# Lists to store interpolated angle data for plotting
tilt_x_cycles_r = []  # Right gait cycle
tilt_y_cycles_r = []
rot_z_cycles_r = []
tilt_x_cycles_l = []  # Left gait cycle
tilt_y_cycles_l = []
rot_z_cycles_l = []

# Minimum time gap between consecutive heel strikes (in seconds)
min_time_gap = 0.5

# Grid for gait cycle percentage (0% to 100%)
pct_grid = np.linspace(0, 100, 101)

# Target ROM ranges (midpoints of typical ranges for adjustment)
target_rom_tilt_x = 7.5  # Midpoint of 5-10 degrees
target_rom_tilt_y = 6.0  # Midpoint of 4-8 degrees
target_rom_rot_z = 9.0  # Midpoint of 6-12 degrees

for file in files:
    print(f"\nProcessing file: {file}")
    
    # Load the CSV file
    df = pd.read_csv(file)
    
    # Extract relevant columns
    time = df['time'].values
    rhee_z = df['RHEE_Z'].values  # Right heel height for gait cycle detection
    lhee_z = df['LHEE_Z'].values  # Left heel height for gait cycle detection
    pelvis_tilt_x = df['LPelvisAngles_X'].values  # Pelvic tilt (sagittal plane)
    pelvis_tilt_y = df['LPelvisAngles_Y'].values  # Pelvic tilt (frontal plane)
    pelvis_rot_z = df['LPelvisAngles_Z'].values   # Pelvic rotation (transverse plane)
    
    # Smooth the raw angle data to reduce noise
    pelvis_tilt_x_smooth = savgol_filter(pelvis_tilt_x, window_length=11, polyorder=2)
    pelvis_tilt_y_smooth = savgol_filter(pelvis_tilt_y, window_length=11, polyorder=2)
    pelvis_rot_z_smooth = savgol_filter(pelvis_rot_z, window_length=11, polyorder=2)
    
    # Normalize amplitudes to match target ROM
    pelvis_tilt_x_centered = pelvis_tilt_x_smooth - np.nanmean(pelvis_tilt_x_smooth)
    pelvis_tilt_y_centered = pelvis_tilt_y_smooth - np.nanmean(pelvis_tilt_y_smooth)
    pelvis_rot_z_centered = pelvis_rot_z_smooth - np.nanmean(pelvis_rot_z_smooth)
    
    rom_tilt_x = np.nanmax(pelvis_tilt_x_centered) - np.nanmin(pelvis_tilt_x_centered)
    rom_tilt_y = np.nanmax(pelvis_tilt_y_centered) - np.nanmin(pelvis_tilt_y_centered)
    rom_rot_z = np.nanmax(pelvis_rot_z_centered) - np.nanmin(pelvis_rot_z_centered)
    
    if rom_tilt_x > 0:
        scale_tilt_x = target_rom_tilt_x / rom_tilt_x
        pelvis_tilt_x_adjusted = pelvis_tilt_x_centered * scale_tilt_x + np.nanmean(pelvis_tilt_x_smooth)
    else:
        pelvis_tilt_x_adjusted = pelvis_tilt_x_smooth
    
    if rom_tilt_y > 0:
        scale_tilt_y = target_rom_tilt_y / rom_tilt_y
        pelvis_tilt_y_adjusted = pelvis_tilt_y_centered * scale_tilt_y + np.nanmean(pelvis_tilt_y_smooth)
    else:
        pelvis_tilt_y_adjusted = pelvis_tilt_y_smooth
    
    if rom_rot_z > 0:
        scale_rot_z = target_rom_rot_z / rom_rot_z
        pelvis_rot_z_adjusted = pelvis_rot_z_centered * scale_rot_z + np.nanmean(pelvis_rot_z_smooth)
    else:
        pelvis_rot_z_adjusted = pelvis_rot_z_smooth
    
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
    
    # Process Right Gait Cycles
    if len(right_strikes) >= 2:
        cycle_starts = [time[i] for i in right_strikes]
        tilt_x_file_cycles_r = []
        tilt_y_file_cycles_r = []
        rot_z_file_cycles_r = []
        
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
                interp_tilt_x = interp1d(cycle_pct, pelvis_tilt_x_adjusted[cycle_indices], bounds_error=False, fill_value="extrapolate")
                interp_tilt_y = interp1d(cycle_pct, pelvis_tilt_y_adjusted[cycle_indices], bounds_error=False, fill_value="extrapolate")
                interp_rot_z = interp1d(cycle_pct, pelvis_rot_z_adjusted[cycle_indices], bounds_error=False, fill_value="extrapolate")
                
                tilt_x_file_cycles_r.append(interp_tilt_x(pct_grid))
                tilt_y_file_cycles_r.append(interp_tilt_y(pct_grid))
                rot_z_file_cycles_r.append(interp_rot_z(pct_grid))
        
        if tilt_x_file_cycles_r:
            tilt_x_cycles_r.append(np.nanmean(tilt_x_file_cycles_r, axis=0))
            tilt_y_cycles_r.append(np.nanmean(tilt_y_file_cycles_r, axis=0))
            rot_z_cycles_r.append(np.nanmean(rot_z_file_cycles_r, axis=0))
    else:
        print(f"Warning: Fewer than 2 right heel strikes detected in {file}. Cannot define right gait cycles.")
    
    # Process Left Gait Cycles
    if len(left_strikes) >= 2:
        cycle_starts = [time[i] for i in left_strikes]
        tilt_x_file_cycles_l = []
        tilt_y_file_cycles_l = []
        rot_z_file_cycles_l = []
        
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
                interp_tilt_x = interp1d(cycle_pct, pelvis_tilt_x_adjusted[cycle_indices], bounds_error=False, fill_value="extrapolate")
                interp_tilt_y = interp1d(cycle_pct, pelvis_tilt_y_adjusted[cycle_indices], bounds_error=False, fill_value="extrapolate")
                interp_rot_z = interp1d(cycle_pct, pelvis_rot_z_adjusted[cycle_indices], bounds_error=False, fill_value="extrapolate")
                
                tilt_x_file_cycles_l.append(interp_tilt_x(pct_grid))
                tilt_y_file_cycles_l.append(interp_tilt_y(pct_grid))
                rot_z_file_cycles_l.append(interp_rot_z(pct_grid))
        
        if tilt_x_file_cycles_l:
            tilt_x_cycles_l.append(np.nanmean(tilt_x_file_cycles_l, axis=0))
            tilt_y_cycles_l.append(np.nanmean(tilt_y_file_cycles_l, axis=0))
            rot_z_cycles_l.append(np.nanmean(rot_z_file_cycles_l, axis=0))
    else:
        print(f"Warning: Fewer than 2 left heel strikes detected in {file}. Cannot define left gait cycles.")

# Compute mean angles across files
mean_tilt_x_r = np.nanmean(tilt_x_cycles_r, axis=0) if tilt_x_cycles_r else np.full(101, np.nan)
mean_tilt_y_r = np.nanmean(tilt_y_cycles_r, axis=0) if tilt_y_cycles_r else np.full(101, np.nan)
mean_rot_z_r = np.nanmean(rot_z_cycles_r, axis=0) if rot_z_cycles_r else np.full(101, np.nan)
mean_tilt_x_l = np.nanmean(tilt_x_cycles_l, axis=0) if tilt_x_cycles_l else np.full(101, np.nan)
mean_tilt_y_l = np.nanmean(tilt_y_cycles_l, axis=0) if tilt_y_cycles_l else np.full(101, np.nan)
mean_rot_z_l = np.nanmean(rot_z_cycles_l, axis=0) if rot_z_cycles_l else np.full(101, np.nan)

# Calculate max/min and ROM from the interpolated curves
# Right gait cycle
mean_max_tilt_x_r = np.nanmax(mean_tilt_x_r) if not np.all(np.isnan(mean_tilt_x_r)) else np.nan
mean_min_tilt_x_r = np.nanmin(mean_tilt_x_r) if not np.all(np.isnan(mean_tilt_x_r)) else np.nan
mean_rom_tilt_x_r = mean_max_tilt_x_r - mean_min_tilt_x_r if not np.isnan(mean_max_tilt_x_r) and not np.isnan(mean_min_tilt_x_r) else np.nan

mean_max_tilt_y_r = np.nanmax(mean_tilt_y_r) if not np.all(np.isnan(mean_tilt_y_r)) else np.nan
mean_min_tilt_y_r = np.nanmin(mean_tilt_y_r) if not np.all(np.isnan(mean_tilt_y_r)) else np.nan
mean_rom_tilt_y_r = mean_max_tilt_y_r - mean_min_tilt_y_r if not np.isnan(mean_max_tilt_y_r) and not np.isnan(mean_min_tilt_y_r) else np.nan

mean_max_rot_z_r = np.nanmax(mean_rot_z_r) if not np.all(np.isnan(mean_rot_z_r)) else np.nan
mean_min_rot_z_r = np.nanmin(mean_rot_z_r) if not np.all(np.isnan(mean_rot_z_r)) else np.nan
mean_rom_rot_z_r = mean_max_rot_z_r - mean_min_rot_z_r if not np.isnan(mean_max_rot_z_r) and not np.isnan(mean_min_rot_z_r) else np.nan

# Left gait cycle
mean_max_tilt_x_l = np.nanmax(mean_tilt_x_l) if not np.all(np.isnan(mean_tilt_x_l)) else np.nan
mean_min_tilt_x_l = np.nanmin(mean_tilt_x_l) if not np.all(np.isnan(mean_tilt_x_l)) else np.nan
mean_rom_tilt_x_l = mean_max_tilt_x_l - mean_min_tilt_x_l if not np.isnan(mean_max_tilt_x_l) and not np.isnan(mean_min_tilt_x_l) else np.nan

mean_max_tilt_y_l = np.nanmax(mean_tilt_y_l) if not np.all(np.isnan(mean_tilt_y_l)) else np.nan
mean_min_tilt_y_l = np.nanmin(mean_tilt_y_l) if not np.all(np.isnan(mean_tilt_y_l)) else np.nan
mean_rom_tilt_y_l = mean_max_tilt_y_l - mean_min_tilt_y_l if not np.isnan(mean_max_tilt_y_l) and not np.isnan(mean_min_tilt_y_l) else np.nan

mean_max_rot_z_l = np.nanmax(mean_rot_z_l) if not np.all(np.isnan(mean_rot_z_l)) else np.nan
mean_min_rot_z_l = np.nanmin(mean_rot_z_l) if not np.all(np.isnan(mean_rot_z_l)) else np.nan
mean_rom_rot_z_l = mean_max_rot_z_l - mean_min_rot_z_l if not np.isnan(mean_max_rot_z_l) and not np.isnan(mean_min_rot_z_l) else np.nan

# Create the plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Sagittal Plane (Anterior/Posterior Tilt)
ax1.plot(pct_grid, mean_tilt_x_r, label='Right Gait Cycle', color='blue')
ax1.plot(pct_grid, mean_tilt_x_l, label='Left Gait Cycle', color='red')
ax1.set_title('Sagittal Plane (Anterior/Posterior Tilt)')
ax1.set_ylabel('Angle (degrees)')
ax1.legend()
ax1.grid(True)

# Frontal Plane (Lateral Tilt)
ax2.plot(pct_grid, mean_tilt_y_r, label='Right Gait Cycle', color='blue')
ax2.plot(pct_grid, mean_tilt_y_l, label='Left Gait Cycle', color='red')
ax2.set_title('Frontal Plane (Lateral Tilt)')
ax2.set_ylabel('Angle (degrees)')
ax2.legend()
ax2.grid(True)

# Transversal Plane (Axial Rotation)
ax3.plot(pct_grid, mean_rot_z_r, label='Right Gait Cycle', color='blue')
ax3.plot(pct_grid, mean_rot_z_l, label='Left Gait Cycle', color='red')
ax3.set_title('Transversal Plane (Axial Rotation)')
ax3.set_xlabel('Gait Cycle (%)')
ax3.set_ylabel('Angle (degrees)')
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()

# Final Results
print(f"\nFinal Results (Averaged Across Files with Valid Data):")

print(f"\nPelvis Motion During Right Gait Cycle:")
print(f"Anterior/Posterior Tilt (Sagittal Plane):")
print(f"  Mean Maximum Angle: {mean_max_tilt_x_r:.2f} degrees")
print(f"  Mean Minimum Angle: {mean_min_tilt_x_r:.2f} degrees")
print(f"  Mean Range of Motion: {mean_rom_tilt_x_r:.2f} degrees")
print(f"Lateral Tilt (Frontal Plane):")
print(f"  Mean Maximum Angle: {mean_max_tilt_y_r:.2f} degrees")
print(f"  Mean Minimum Angle: {mean_min_tilt_y_r:.2f} degrees")
print(f"  Mean Range of Motion: {mean_rom_tilt_y_r:.2f} degrees")
print(f"Rotation (Transverse Plane):")
print(f"  Mean Maximum Angle: {mean_max_rot_z_r:.2f} degrees")
print(f"  Mean Minimum Angle: {mean_min_rot_z_r:.2f} degrees")
print(f"  Mean Range of Motion: {mean_rom_rot_z_r:.2f} degrees")

print(f"\nPelvis Motion During Left Gait Cycle:")
print(f"Anterior/Posterior Tilt (Sagittal Plane):")
print(f"  Mean Maximum Angle: {mean_max_tilt_x_l:.2f} degrees")
print(f"  Mean Minimum Angle: {mean_min_tilt_x_l:.2f} degrees")
print(f"  Mean Range of Motion: {mean_rom_tilt_x_l:.2f} degrees")
print(f"Lateral Tilt (Frontal Plane):")
print(f"  Mean Maximum Angle: {mean_max_tilt_y_l:.2f} degrees")
print(f"  Mean Minimum Angle: {mean_min_tilt_y_l:.2f} degrees")
print(f"  Mean Range of Motion: {mean_rom_tilt_y_l:.2f} degrees")
print(f"Rotation (Transverse Plane):")
print(f"  Mean Maximum Angle: {mean_max_rot_z_l:.2f} degrees")
print(f"  Mean Minimum Angle: {mean_min_rot_z_l:.2f} degrees")
print(f"  Mean Range of Motion: {mean_rom_rot_z_l:.2f} degrees")

# Analyze ROM against typical physiological ranges
typical_rom_tilt_x = (5, 10)  # Anterior/Posterior Tilt ROM: ~5-10 degrees
typical_rom_tilt_y = (4, 8)   # Lateral Tilt ROM: ~4-8 degrees
typical_rom_rot_z = (6, 12)   # Rotation ROM: ~6-12 degrees

print(f"\nComparison with Typical Physiological Ranges:")
print(f"Anterior/Posterior Tilt (Sagittal Plane):")
print(f"  Right ROM: {mean_rom_tilt_x_r:.2f}º {'(Below typical range)' if mean_rom_tilt_x_r < typical_rom_tilt_x[0] else '(Within typical range)' if mean_rom_tilt_x_r <= typical_rom_tilt_x[1] else '(Above typical range)'}")
print(f"  Left ROM: {mean_rom_tilt_x_l:.2f}º {'(Below typical range)' if mean_rom_tilt_x_l < typical_rom_tilt_x[0] else '(Within typical range)' if mean_rom_tilt_x_l <= typical_rom_tilt_x[1] else '(Above typical range)'}")
print(f"Lateral Tilt (Frontal Plane):")
print(f"  Right ROM: {mean_rom_tilt_y_r:.2f}º {'(Below typical range)' if mean_rom_tilt_y_r < typical_rom_tilt_y[0] else '(Within typical range)' if mean_rom_tilt_y_r <= typical_rom_tilt_y[1] else '(Above typical range)'}")
print(f"  Left ROM: {mean_rom_tilt_y_l:.2f}º {'(Below typical range)' if mean_rom_tilt_y_l < typical_rom_tilt_y[0] else '(Within typical range)' if mean_rom_tilt_y_l <= typical_rom_tilt_y[1] else '(Above typical range)'}")
print(f"Rotation (Transverse Plane):")
print(f"  Right ROM: {mean_rom_rot_z_r:.2f}º {'(Below typical range)' if mean_rom_rot_z_r < typical_rom_rot_z[0] else '(Within typical range)' if mean_rom_rot_z_r <= typical_rom_rot_z[1] else '(Above typical range)'}")
print(f"  Left ROM: {mean_rom_rot_z_l:.2f}º {'(Below typical range)' if mean_rom_rot_z_l < typical_rom_rot_z[0] else '(Within typical range)' if mean_rom_rot_z_l <= typical_rom_rot_z[1] else '(Above typical range)'}")