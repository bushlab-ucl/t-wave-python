# %% IMPORTS

import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import Simulations
from Simulations import PhaseTrackerStatus
import os
from scipy import stats

from analyze_time_frequency import get_p_c_struct, filter_for_competing_events, find_minima, find_maxima
from load_intracranial_data import load_data_as_dataset


# %% LOAD DATA

patient03_channel1_eeg = np.load("data/annotated/Patient03_Channel1_EEG.npy")
patient03_channel1_sws = np.load("data/annotated/Patient03_Channel1_negSWs.npy")
patient03_channel1_ieds = np.load("data/annotated/Patient03_Channel1_IEDs.npy")

with open("results/results_zerocross_patient03_channel1_08_newbackoff_sp.pkl", "rb") as f:
    results_twave = pickle.load(f)

# %% COMPUTE MEAN ABSOLUTE AMPLITUDE AROUND GROUND TRUTH SLOW WAVE VS NON-SLOW-WAVE

def index_mask(slow_wave_idx, signal_length, window=512):
    half = window // 2
    mask = np.zeros(signal_length, dtype=bool)

    for idx in slow_wave_idx:
        start = max(0, idx - half)
        end = min(signal_length, idx + half)
        mask[int(start):int(end)] = True

    return mask

sw_idx = (patient03_channel1_sws*512).astype(int)
ied_idx = (patient03_channel1_ieds*512).astype(int)

sw_idx_mask = index_mask(sw_idx, len(patient03_channel1_eeg), window=512)
ied_idx_mask = index_mask(ied_idx, len(patient03_channel1_eeg), window=512)

eeg = patient03_channel1_eeg.squeeze()

both_idx_mask = sw_idx_mask + ied_idx_mask

mean_amp_sw = np.mean(np.abs(eeg[sw_idx_mask]))
mean_amp_ied = np.mean(np.abs(eeg[ied_idx_mask]))
mean_amp_non_sw = np.mean(np.abs(eeg[~both_idx_mask]))

median_amp_sw = np.median(np.abs(eeg[sw_idx_mask]))
median_amp_ied = np.median(np.abs(eeg[ied_idx_mask]))
median_amp_non_sw = np.median(np.abs(eeg[~both_idx_mask]))

print(f"mean_amp_sw: {mean_amp_sw}")
print(f"mean_amp_ied: {mean_amp_ied}")
print(f"mean_amp_non_sw: {mean_amp_non_sw}")

print(f"median_amp_sw: {median_amp_sw}")
print(f"median_amp_ied: {median_amp_ied}")
print(f"median_amp_non_sw: {median_amp_non_sw}")

# %% PLOT DISTRIBUTION OF AMPLITUDES INSIDE SLOW WAVE WINDOW

plt.figure(figsize=(8, 5))

plt.hist(eeg[~sw_idx_mask], bins=100, alpha=0.5, density=True, label="non-slow-wave amplitudes")
plt.hist(eeg[sw_idx_mask], bins=100, alpha=0.5, density=True, label="slow wave amplitudes")
plt.hist(eeg[ied_idx_mask], bins=100, alpha=0.5, density=True, label="ied amplitudes")

plt.axvline(mean_amp_sw, color="orange", label="mean_amp_sw", ymin=0, ymax=1)
plt.axvline(mean_amp_ied, color="green", label="mean_amp_ied", ymin=0, ymax=1)
plt.axvline(mean_amp_non_sw, color="blue", label="mean_amp_non_sw", ymin=0, ymax=1)

plt.xlabel("amplitude (microV)")
plt.ylabel("counts")
plt.xlim(-5000, 5000)
plt.grid(True, alpha=0.3, axis="y")
plt.legend()
plt.tight_layout()
plt.show()

# %% PLOT PHASE ESTIMATE HISTOGRAM AT STIMULATION INDICES

results_twave.plot_phase_hist()

# %% PLOT PHASE ESTIMATE HISTOGRAM AT GROUND TRUTH SLOW WAVE INDICES

sw_idx_trunc = np.array([i for i in sw_idx if i/results_twave.Dataset.fs <= results_twave.Dataset.t.max()])

internals = results_twave.internals_ts

sw_phases = np.array([
    internals[int(i)]["phase"]
    for i in sw_idx_trunc
])

print(sw_phases)

Simulations.plot_phase_hist_array(stim_phases=sw_phases, target_phase=0, title="Phase estimate distribution at ground-truth slow waves")

# %% COMPUTE AMOUNTS OF DIFFERENT PHASETRACKER STATUSES

status = np.array(results_twave.status_ts)

def count(flag):
    return np.sum((status & flag) > 0)

print("NONE:", count(PhaseTrackerStatus.NONE))

print("STIM1:", count(PhaseTrackerStatus.STIM1))
print("STIM2:", count(PhaseTrackerStatus.STIM2))

print("INHIBITED:", count(PhaseTrackerStatus.INHIBITED))
print("INHIBITED_AMP:", count(PhaseTrackerStatus.INHIBITED_AMP))
print("INHIBITED_RATIO:", count(PhaseTrackerStatus.INHIBITED_RATIO))
print("INHIBITED_QUADRATURE:", count(PhaseTrackerStatus.INHIBITED_QUADRATURE))

print("WRONGPHASE:", count(PhaseTrackerStatus.WRONGPHASE))
print("BACKOFF:", count(PhaseTrackerStatus.BACKOFF))
print("BACKOFF_ISI:", count(PhaseTrackerStatus.BACKOFF_ISI))

# %% PRINT PHASETRACKER STATUS AT GROUND TRUTH SLOW WAVE INDICES

print(len(results_twave.status_ts))
print(PhaseTrackerStatus.STIM1)

path_sw = Path("data/annotated/Patient03_Channel1_negSWs.npy")
path_ied = Path("data/annotated/Patient03_Channel1_IEDs.npy")

arr_sw = np.load(path_sw)
arr_ied = np.load(path_ied)

arr_sw_trunc = np.array([x * results_twave.Dataset.fs for x in arr_sw if x <= results_twave.Dataset.t.max()], dtype=int)
arr_ied_trunc = np.array([x * results_twave.Dataset.fs for x in arr_ied if x <= results_twave.Dataset.t.max()], dtype=int)


print(results_twave.status_ts[:50])

for i in arr_sw_trunc:
    print(results_twave.status_ts[int(i)].name)
    
# %% INVESTIGATE hl_ratio

#print(results_twave.internals_ts)
internals = results_twave.internals_ts

hl_ratios = []
for i in range(len(internals)):
    hl_ratios.append(internals[i]["hl_ratio"])
hl_ratios = np.array(hl_ratios)
print(len(hl_ratios))
print(hl_ratios[:20])

print(np.sum(hl_ratios))
# %% ANALYZE MINIMAL AMPLITUDE DISTRIBUTION OF IEDs vs SWs

data_dir = "data/annotated"

p_c_struct = get_p_c_struct(data_dir)
min_amps_sw = []
min_amps_ied = []


for p, cs in p_c_struct.items():

    for c in cs:
       
        # load data
        signal_filename = f"Patient{p}_Channel{c}_EEG.npy"
        signal_filepath = os.path.join(data_dir, signal_filename)

        sws_filename = f"Patient{p}_Channel{c}_negSWs.npy"
        sws_filepath = os.path.join(data_dir, sws_filename)

        ieds_filename = f"Patient{p}_Channel{c}_IEDs.npy"
        ieds_filepath = os.path.join(data_dir, ieds_filename)

        dataset = load_data_as_dataset(signal_filepath, fs=512)

        signal = dataset.raw_signal

        arr_sws = np.load(sws_filepath)
        arr_ieds = np.load(ieds_filepath)

        # filter for competing events
        filtered_arr_ied, filtered_arr_sw = filter_for_competing_events(arr_ieds, arr_sws, buffer_s=4)
        
        # find minima of sws
        sw_minima = find_minima(signal, filtered_arr_sw, fs=512, window_size_s=4, low_pass=4)

        # find minima of ieds
        ied_minima = find_minima(signal, filtered_arr_ied, fs=512, window_size_s=4, low_pass=4)

        # compute amplitudes at minima
        this_min_amps_sw = signal[sw_minima]
        this_min_amps_ied = signal[ied_minima]

        # extend list 
        min_amps_sw.extend(this_min_amps_sw)
        min_amps_ied.extend(this_min_amps_ied)

mean_min_amp_sw = np.mean(min_amps_sw)
mean_min_amp_ied = np.mean(min_amps_ied)

print(mean_min_amp_sw, mean_min_amp_ied)

plt.hist(min_amps_ied, bins=50, alpha=0.3, density=True, label="IED pos amps")
plt.hist(min_amps_sw, bins=50, alpha=0.3, density=True, label="SW pos amps")
plt.legend()
plt.show()

p_range = [20, 10, 5, 1, 0.1]
sw_percentiles = np.nanpercentile(min_amps_sw, p_range)
ied_percentiles = np.nanpercentile(min_amps_ied, p_range)

# Print the results in a readable format
print(f"{'Percentile':<12} | {'SW':<10} | {'IED':<10}")
print("-" * 35)
for p, sw_val, ied_val in zip(p_range, sw_percentiles, ied_percentiles):
    print(f"{p:<12} | {sw_val:>10.2f} | {ied_val:>10.2f}")

# Print the results in a readable format
print(f"{'Percentile':<12} | {'SW':<10} | {'IED':<10}")
print("-" * 35)
for p, sw_val, ied_val in zip(p_range, sw_percentiles, ied_percentiles):
    print(f"{p:<12} | {sw_val:>10.2f} | {ied_val:>10.2f}")

# Define the specific amplitudes you want to check
amp_values = [-5000, -4000, -3000, -2000]

# Convert lists to numpy arrays
sw_array = np.array(min_amps_sw)
ied_array = np.array(min_amps_ied)

# Use the arrays for the masking operation
sw_clean = sw_array[~np.isnan(sw_array)]
ied_clean = ied_array[~np.isnan(ied_array)]

# Print the results in the same structure as before
print(f"{'Amplitude':<12} | {'SW %-tile':<10} | {'IED %-tile':<10}")
print("-" * 38)

for val in amp_values:
    # kind='rank' gives the percentage of values less than or equal to the score
    sw_p = stats.percentileofscore(sw_clean, val, kind='rank')
    ied_p = stats.percentileofscore(ied_clean, val, kind='rank')
    
    print(f"{val:<12} | {sw_p:>10.2f}% | {ied_p:>10.2f}%")

# %% ANALYZE MAXIMAL AMPLITUDE DISTRIBUTION OF IEDs vs SWs

data_dir = "data/annotated"

p_c_struct = get_p_c_struct(data_dir)
max_amps_sw = []
max_amps_ied = []


for p, cs in p_c_struct.items():

    for c in cs:
       
        # load data
        signal_filename = f"Patient{p}_Channel{c}_EEG.npy"
        signal_filepath = os.path.join(data_dir, signal_filename)

        sws_filename = f"Patient{p}_Channel{c}_negSWs.npy"
        sws_filepath = os.path.join(data_dir, sws_filename)

        ieds_filename = f"Patient{p}_Channel{c}_IEDs.npy"
        ieds_filepath = os.path.join(data_dir, ieds_filename)

        dataset = load_data_as_dataset(signal_filepath, fs=512)

        signal = dataset.raw_signal

        arr_sws = np.load(sws_filepath)
        arr_ieds = np.load(ieds_filepath)

        # filter for competing events
        filtered_arr_ied, filtered_arr_sw = filter_for_competing_events(arr_ieds, arr_sws, buffer_s=4)
        
        # find maxima of sws
        sw_maxima = find_maxima(signal, filtered_arr_sw, fs=512, window_size_s=4, low_pass=4)

        # find maxima of ieds
        ied_maxima = find_maxima(signal, filtered_arr_ied, fs=512, window_size_s=4, low_pass=4)

        # compute amplitudes at maxima
        this_max_amps_sw = signal[sw_maxima]
        this_max_amps_ied = signal[ied_maxima]

        # extend list 
        max_amps_sw.extend(this_max_amps_sw)
        max_amps_ied.extend(this_max_amps_ied)

mean_max_amp_sw = np.mean(max_amps_sw)
mean_max_amp_ied = np.mean(max_amps_ied)

print(mean_max_amp_sw, mean_max_amp_ied)

plt.hist(max_amps_ied, bins=50, alpha=0.3, density=True, label="IED pos amps")
plt.hist(max_amps_sw, bins=50, alpha=0.3, density=True, label="SW pos amps")
plt.legend()
plt.show()

p_range = [90, 95, 99, 99.9]
sw_percentiles = np.nanpercentile(max_amps_sw, p_range)
ied_percentiles = np.nanpercentile(max_amps_ied, p_range)

# Print the results in a readable format
print(f"{'Percentile':<12} | {'SW':<10} | {'IED':<10}")
print("-" * 35)
for p, sw_val, ied_val in zip(p_range, sw_percentiles, ied_percentiles):
    print(f"{p:<12} | {sw_val:>10.2f} | {ied_val:>10.2f}")

# Define the specific amplitudes you want to check
amp_values = [2000, 3000, 4000, 5000]

# Convert lists to numpy arrays
sw_array = np.array(max_amps_sw)
ied_array = np.array(max_amps_ied)

# Use the arrays for the masking operation
sw_clean = sw_array[~np.isnan(sw_array)]
ied_clean = ied_array[~np.isnan(ied_array)]

# Print the results in the same structure as before
print(f"{'Amplitude':<12} | {'SW %-tile':<10} | {'IED %-tile':<10}")
print("-" * 38)

for val in amp_values:
    # kind='rank' gives the percentage of values less than or equal to the score
    sw_p = stats.percentileofscore(sw_clean, val, kind='rank')
    ied_p = stats.percentileofscore(ied_clean, val, kind='rank')
    
    print(f"{val:<12} | {sw_p:>10.2f}% | {ied_p:>10.2f}%")

# %%
