import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter  # For smoothing the data

# List of file numbers
file_nums = range(16, 21)
files = [f'S13_D1_shoe_normal_rd1_{num}.csv' for num in file_nums]

# Initialize lists for gait cycle duration and height profiles (Script 1)
right_cycle_durations = []
left_cycle_durations = []
right_avg_heights = []  # For heel heights
left_avg_heights = []
right_toe_avg_heights = []  # For toe heights
left_toe_avg_heights = []

# Initialize lists for stance and swing phase durations (Script 2)
right_stance_durations = []
right_swing_durations = []
left_stance_durations = []
left_swing_durations = []

# Initialize lists for cadence and step frequency (Script 3)
cadences = []
step_frequencies = []

# Minimum time gaps (in seconds)
min_time_gap = 0.5  # Between consecutive heel strikes
min_stance_duration = 0.4  # Between heel strike and toe-off

for file in files:
    print(f"\nProcessing file: {file}")

    # Load the CSV file
    df = pd.read_csv(file)

    # Extract relevant columns
    time = df['time'].values
    rhee_z = df['RHEE_Z'].values  # Right heel height
    lhee_z = df['LHEE_Z'].values  # Left heel height
    rtoe_z = df['RTOE_Z'].values  # Right toe height
    ltoe_z = df['LTOE_Z'].values  # Left toe height

    # Smooth the height data to reduce noise
    rhee_z_smooth = savgol_filter(rhee_z, window_length=11, polyorder=2)
    lhee_z_smooth = savgol_filter(lhee_z, window_length=11, polyorder=2)
    rtoe_z_smooth = savgol_filter(rtoe_z, window_length=11, polyorder=2)
    ltoe_z_smooth = savgol_filter(ltoe_z, window_length=11, polyorder=2)

    # Debug: Check min and max heights
    print(f"Right heel (RHEE_Z) min: {np.min(rhee_z_smooth):.2f}, max: {np.max(rhee_z_smooth):.2f}")
    print(f"Left heel (LHEE_Z) min: {np.min(lhee_z_smooth):.2f}, max: {np.max(lhee_z_smooth):.2f}")
    print(f"Right toe (RTOE_Z) min: {np.min(rtoe_z_smooth):.2f}, max: {np.max(rtoe_z_smooth):.2f}")
    print(f"Left toe (LTOE_Z) min: {np.min(ltoe_z_smooth):.2f}, max: {np.max(ltoe_z_smooth):.2f}")

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

    # Debug: Print detected heel strikes
    print(f"Right foot strikes: {len(right_strikes)} at times {time[right_strikes] if right_strikes else 'None'}")
    print(f"Left foot strikes: {len(left_strikes)} at times {time[left_strikes] if left_strikes else 'None'}")

    # --- Script 1: Gait Cycle Duration and Height Profiles ---
    # Calculate gait cycle durations
    if len(right_strikes) > 1:
        right_cycle_times = np.diff(time[right_strikes])
        right_cycle_durations.extend(right_cycle_times)
    if len(left_strikes) > 1:
        left_cycle_times = np.diff(time[left_strikes])
        left_cycle_durations.extend(left_cycle_times)

    # Extract the first full cycle for averaging the height profile (Heels)
    if len(right_strikes) > 1:
        start_idx = right_strikes[0]
        end_idx = right_strikes[1]
        right_cycle = rhee_z[start_idx:end_idx+1]
        right_cycle_resampled = np.interp(np.linspace(0, len(right_cycle)-1, 100), 
                                         np.arange(len(right_cycle)), right_cycle)
        right_avg_heights.append(right_cycle_resampled)

    if len(left_strikes) > 1:
        start_idx = left_strikes[0]
        end_idx = left_strikes[1]
        left_cycle = lhee_z[start_idx:end_idx+1]
        left_cycle_resampled = np.interp(np.linspace(0, len(left_cycle)-1, 100), 
                                        np.arange(len(left_cycle)), left_cycle)
        left_avg_heights.append(left_cycle_resampled)

    # Extract the first full cycle for averaging the toe height profile (Toes)
    if len(right_strikes) > 1:
        start_idx = right_strikes[0]
        end_idx = right_strikes[1]
        right_toe_cycle = rtoe_z[start_idx:end_idx+1]
        right_toe_cycle_resampled = np.interp(np.linspace(0, len(right_toe_cycle)-1, 100), 
                                             np.arange(len(right_toe_cycle)), right_toe_cycle)
        right_toe_avg_heights.append(right_toe_cycle_resampled)

    if len(left_strikes) > 1:
        start_idx = left_strikes[0]
        end_idx = left_strikes[1]
        left_toe_cycle = ltoe_z[start_idx:end_idx+1]
        left_toe_cycle_resampled = np.interp(np.linspace(0, len(left_toe_cycle)-1, 100), 
                                            np.arange(len(left_toe_cycle)), left_toe_cycle)
        left_toe_avg_heights.append(left_toe_cycle_resampled)

    # --- Script 2: Stance and Swing Phase Durations ---
    # Detect toe-off events by targeting 60% of the gait cycle
    right_toe_offs = []
    for i in range(len(right_strikes) - 1):
        current_strike = right_strikes[i]
        next_strike = right_strikes[i + 1]
        cycle_duration = time[next_strike] - time[current_strike]
        
        # Calculate the expected toe-off time (60% of the gait cycle)
        expected_toe_off_time = time[current_strike] + 0.6 * cycle_duration
        
        # Define a search window around the expected toe-off time (e.g., 50% to 70% of the cycle)
        window_start_time = time[current_strike] + 0.5 * cycle_duration
        window_end_time = time[current_strike] + 0.7 * cycle_duration
        window_start_idx = np.searchsorted(time, window_start_time, side='left')
        window_end_idx = np.searchsorted(time, window_end_time, side='right')
        
        # Find the point of maximum toe height increase within the window
        max_slope = 0
        toe_off_idx = None
        for j in range(max(window_start_idx, 1), min(window_end_idx, len(rtoe_z_smooth) - 1)):
            slope = (rtoe_z_smooth[j] - rtoe_z_smooth[j-1]) / (time[j] - time[j-1])
            if slope > max_slope:
                max_slope = slope
                toe_off_idx = j
        
        if toe_off_idx and (time[toe_off_idx] - time[current_strike]) >= min_stance_duration:
            right_toe_offs.append(toe_off_idx)
            # Debug: Calculate the percentage of the gait cycle for this toe-off
            toe_off_percentage = ((time[toe_off_idx] - time[current_strike]) / cycle_duration) * 100
            print(f"Right toe-off at {time[toe_off_idx]:.2f}s, {toe_off_percentage:.2f}% of cycle")

    left_toe_offs = []
    for i in range(len(left_strikes) - 1):
        current_strike = left_strikes[i]
        next_strike = left_strikes[i + 1]
        cycle_duration = time[next_strike] - time[current_strike]
        
        # Calculate the expected toe-off time (60% of the gait cycle)
        expected_toe_off_time = time[current_strike] + 0.6 * cycle_duration
        
        # Define a search window around the expected toe-off time (e.g., 50% to 70% of the cycle)
        window_start_time = time[current_strike] + 0.5 * cycle_duration
        window_end_time = time[current_strike] + 0.7 * cycle_duration
        window_start_idx = np.searchsorted(time, window_start_time, side='left')
        window_end_idx = np.searchsorted(time, window_end_time, side='right')
        
        # Find the point of maximum toe height increase within the window
        max_slope = 0
        toe_off_idx = None
        for j in range(max(window_start_idx, 1), min(window_end_idx, len(ltoe_z_smooth) - 1)):
            slope = (ltoe_z_smooth[j] - ltoe_z_smooth[j-1]) / (time[j] - time[j-1])
            if slope > max_slope:
                max_slope = slope
                toe_off_idx = j
        
        if toe_off_idx and (time[toe_off_idx] - time[current_strike]) >= min_stance_duration:
            left_toe_offs.append(toe_off_idx)
            # Debug: Calculate the percentage of the gait cycle for this toe-off
            toe_off_percentage = ((time[toe_off_idx] - time[current_strike]) / cycle_duration) * 100
            print(f"Left toe-off at {time[toe_off_idx]:.2f}s, {toe_off_percentage:.2f}% of cycle")

    # Debug: Print detected toe-offs
    print(f"Right foot toe-offs: {len(right_toe_offs)} at times {time[right_toe_offs] if right_toe_offs else 'None'}")
    print(f"Left foot toe-offs: {len(left_toe_offs)} at times {time[left_toe_offs] if left_toe_offs else 'None'}")

    # Calculate stance and swing phase durations for the right foot
    for i in range(len(right_strikes) - 1):
        current_strike = right_strikes[i]
        next_strike = right_strikes[i + 1]
        toe_off = [to for to in right_toe_offs if current_strike < to < next_strike]
        if toe_off:
            toe_off = toe_off[0]
            stance_duration = time[toe_off] - time[current_strike]
            right_stance_durations.append(stance_duration)
            swing_duration = time[next_strike] - time[toe_off]
            right_swing_durations.append(swing_duration)

    # Calculate stance and swing phase durations for the left foot
    for i in range(len(left_strikes) - 1):
        current_strike = left_strikes[i]
        next_strike = left_strikes[i + 1]
        toe_off = [to for to in left_toe_offs if current_strike < to < next_strike]
        if toe_off:
            toe_off = toe_off[0]
            stance_duration = time[toe_off] - time[current_strike]
            left_stance_durations.append(stance_duration)
            swing_duration = time[next_strike] - time[toe_off]
            left_swing_durations.append(swing_duration)

    # --- Script 3: Cadence and Step Frequency ---
    # Combine strikes for this file into a single list of (time, foot) tuples
    file_strikes = []
    for i in right_strikes:
        file_strikes.append((time[i], 'right'))
    for i in left_strikes:
        file_strikes.append((time[i], 'left'))

    # Sort strikes by time
    file_strikes.sort(key=lambda x: x[0])

    # Debug: Print strikes for this file
    print(f"Strikes for {file} (sorted by time):")
    for strike in file_strikes:
        print(f"Time: {strike[0]:.2f}, Foot: {strike[1]}")

    # Calculate steps (each consecutive pair of strikes from different feet is a step)
    step_durations = []
    for i in range(len(file_strikes) - 1):
        if file_strikes[i][1] != file_strikes[i+1][1]:
            step_duration = file_strikes[i+1][0] - file_strikes[i][0]
            step_durations.append(step_duration)

    # Calculate total number of steps and total time for this file
    total_steps = len(step_durations)
    if file_strikes:
        total_time_seconds = file_strikes[-1][0] - file_strikes[0][0]
        total_time_minutes = total_time_seconds / 60
    else:
        total_time_seconds = 0
        total_time_minutes = 0

    # Calculate cadence for this file
    file_cadence = (total_steps / total_time_minutes) if total_time_minutes > 0 else 0

    # Calculate average step duration and step frequency for this file
    mean_step_duration = np.mean(step_durations) if step_durations else 0
    file_step_frequency = (1 / mean_step_duration) if mean_step_duration > 0 else 0

    # Store results
    cadences.append(file_cadence)
    step_frequencies.append(file_step_frequency)

    # Print results for this file
    print(f"\nResults for {file}:")
    print(f"Total Steps: {total_steps}")
    print(f"Total Time: {total_time_seconds:.2f} seconds ({total_time_minutes:.2f} minutes)")
    print(f"Cadence: {file_cadence:.2f} steps/minute")
    print(f"Average Step Duration: {mean_step_duration:.2f} seconds")
    print(f"Step Frequency (1/Step Duration): {file_step_frequency:.2f} Hz")

# --- Final Results ---
# Script 1: Gait Cycle Duration and Height Profiles
mean_right_cycle_duration = np.mean(right_cycle_durations) if right_cycle_durations else 0
mean_left_cycle_duration = np.mean(left_cycle_durations) if left_cycle_durations else 0
print(f"\nFinal Results:")
print(f"Mean Right Foot Cycle Duration: {mean_right_cycle_duration:.2f} seconds")
print(f"Mean Left Foot Cycle Duration: {mean_left_cycle_duration:.2f} seconds")

# Calculate and plot average heel height profiles
if right_avg_heights:
    right_avg_profile = np.mean(right_avg_heights, axis=0)
else:
    right_avg_profile = np.zeros(100)
if left_avg_heights:
    left_avg_profile = np.mean(left_avg_heights, axis=0)
else:
    left_avg_profile = np.zeros(100)

plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0, 100, 100), right_avg_profile, label='Right Heel (RHEE_Z)', color='blue')
plt.plot(np.linspace(0, 100, 100), left_avg_profile, label='Left Heel (LHEE_Z)', color='red')
plt.xlabel('Gait Cycle (%)')
plt.ylabel('Average Height (z-coordinate, mm)')
plt.title('Average Heel Height Profile of Gait Cycle (Right vs Left Foot)')
plt.legend()
plt.grid(True)
plt.show()

# Calculate and plot average toe height profiles
if right_toe_avg_heights:
    right_toe_avg_profile = np.mean(right_toe_avg_heights, axis=0)
else:
    right_toe_avg_profile = np.zeros(100)
if left_toe_avg_heights:
    left_toe_avg_profile = np.mean(left_toe_avg_heights, axis=0)
else:
    left_toe_avg_profile = np.zeros(100)

plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0, 100, 100), right_toe_avg_profile, label='Right Toe (RTOE_Z)', color='blue')
plt.plot(np.linspace(0, 100, 100), left_toe_avg_profile, label='Left Toe (LTOE_Z)', color='red')
plt.xlabel('Gait Cycle (%)')
plt.ylabel('Average Height (z-coordinate, mm)')
plt.title('Average Toe Height Profile of Gait Cycle (Right vs Left Foot)')
plt.legend()
plt.grid(True)
plt.show()

# Script 2: Stance and Swing Phase Durations
mean_right_stance = np.mean(right_stance_durations) if right_stance_durations else 0
mean_right_swing = np.mean(right_swing_durations) if right_swing_durations else 0
mean_left_stance = np.mean(left_stance_durations) if left_stance_durations else 0
mean_left_swing = np.mean(left_swing_durations) if left_swing_durations else 0

# Print stance and swing phase durations in the requested format
print(f"Final Results:")
print(f"Right Foot Mean Stance Phase Duration: {mean_right_stance:.2f} seconds")
print(f"Right Foot Mean Swing Phase Duration: {mean_right_swing:.2f} seconds")
print(f"Left Foot Mean Stance Phase Duration: {mean_left_stance:.2f} seconds")
print(f"Left Foot Mean Swing Phase Duration: {mean_left_swing:.2f} seconds")

# Additional verification of gait cycle duration
print(f"Right Foot Gait Cycle (Stance + Swing): {mean_right_stance + mean_right_swing:.2f} seconds")
print(f"Left Foot Gait Cycle (Stance + Swing): {mean_left_stance + mean_left_swing:.2f} seconds")

# Verify the percentage of stance and swing phases
right_stance_percentage = (mean_right_stance / (mean_right_stance + mean_right_swing)) * 100 if (mean_right_stance + mean_right_swing) > 0 else 0
right_swing_percentage = (mean_right_swing / (mean_right_stance + mean_right_swing)) * 100 if (mean_right_stance + mean_right_swing) > 0 else 0
left_stance_percentage = (mean_left_stance / (mean_left_stance + mean_left_swing)) * 100 if (mean_left_stance + mean_left_swing) > 0 else 0
left_swing_percentage = (mean_left_swing / (mean_left_stance + mean_left_swing)) * 100 if (mean_left_stance + mean_left_swing) > 0 else 0

print(f"Right Foot Stance Phase: {right_stance_percentage:.2f}%")
print(f"Right Foot Swing Phase: {right_swing_percentage:.2f}%")
print(f"Left Foot Stance Phase: {left_stance_percentage:.2f}%")
print(f"Left Foot Swing Phase: {left_swing_percentage:.2f}%")

# Script 3: Cadence and Step Frequency
mean_cadence = np.mean(cadences) if cadences else 0
mean_step_frequency = np.mean(step_frequencies) if step_frequencies else 0
mean_step_frequency_from_cadence = mean_cadence / 120

print(f"Mean Cadence: {mean_cadence:.2f} steps/minute")
print(f"Mean Step Frequency (1/Step Duration): {mean_step_frequency:.2f} Hz")
print(f"Mean Step Frequency (Cadence/120): {mean_step_frequency_from_cadence:.2f} Hz")