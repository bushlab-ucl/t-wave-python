# %% IMPORTS

import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import Simulations
from Simulations import PhaseTrackerStatus


# %% LOAD DATA

with open("/home/jhedemann/slow-wave/1024hz/Patient04_OfflineMrk.mrk", "r", encoding="utf-8") as input_file:
    text = input_file.read()

patient03_channel1_eeg = np.load("/home/jhedemann/slow-wave/annotated/Patient03_Channel1_EEG.npy")
patient03_channel1_sws = np.load("/home/jhedemann/slow-wave/annotated/Patient03_Channel1_SWs.npy")
patient03_channel1_ieds = np.load("/home/jhedemann/slow-wave/annotated/Patient03_Channel1_IEDs.npy")

with open("results/results_twave_patient03_channel1_12_hl_frequency_changed.pkl", "rb") as f:
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

path_sw = Path("/home/jhedemann/slow-wave/annotated/Patient03_Channel1_SWs.npy")
path_ied = Path("/home/jhedemann/slow-wave/annotated/Patient03_Channel1_IEDs.npy")

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
# %%
