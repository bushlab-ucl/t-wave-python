# %% IMPORTS

import numpy as np
from pathlib import Path
import re
import pickle
from scipy.signal import butter, sosfiltfilt, resample_poly
import matplotlib.pyplot as plt

from load_intracranial_data import load_data_as_dataset, load_sw_annotation
from analyze_time_frequency import get_signal_subsets_from_events

import Simulations
from Algo_ZeroCrossing import PhaseTracker as ZeroCross


# %%

time_excerpt = 0 # seconds
sampling_rate = 512 # hz

# %%

ds = load_data_as_dataset(npy_path="/home/jhedemann/ptas_benchmarks_jhedemann/data/annotated/Patient03_Channel1_EEG.npy",
                          fs=sampling_rate)
if time_excerpt != 0:
    ds_trunc = ds.signal.squeeze().astype(float)[:time_excerpt*sampling_rate]
else:
    ds_trunc = ds.signal.squeeze().astype(float)

sos = butter(4, [0.5, 4.0], btype="bandpass", fs=ds.fs, output="sos")
ds_filtered = sosfiltfilt(sos, ds_trunc)

#%% 

print("ds.fs:", ds.fs)
print("ds.signal.shape:", np.asarray(ds.signal).shape)
print("ds_trunc.shape:", ds_trunc.shape)
print("len(ds_trunc):", len(ds_trunc))
print("duration_s (from ds_trunc):", len(ds_trunc)/ds.fs)

# %%

ds_sim = Simulations.SimulationDataset(
    t=np.arange(len(ds_filtered)) / ds.fs,
    signal=ds_filtered,
    fs=ds.fs,
    name=ds.name + f"_p03-c1_ds{ds.fs}",
)

rslt = Simulations.run_simulations(ds_sim, ZeroCross(fs=ds_sim.fs, amp_q=0.05))

# %%

with open("results/results_zerocross_patient03_channel1_19_newdynamic.pkl", "wb") as f:
    pickle.dump(rslt, f)

# %%

print(rslt.stims_sp)
print(len(rslt.stims_sp))

# %%

stims_mean = np.mean(np.diff(rslt.stims_sp))
stims_std = np.std(np.diff(rslt.stims_sp))
print(stims_mean, stims_std)

# %% 

neg_sw_annot = "/home/jhedemann/ptas_benchmarks_jhedemann/data/annotated/Patient03_Channel1_negSWs.npy"
arr_neg_sw = load_sw_annotation(neg_sw_annot)

neg_sw_windows = get_signal_subsets_from_events(signal=rslt.Dataset.signal,
                                                idx=arr_neg_sw*rslt.Dataset.fs,
                                                fs=rslt.Dataset.fs,
                                                window_size_s=2)

neg_sw_mins = np.array([np.min(window) for window in neg_sw_windows])
print(neg_sw_mins)
print(np.mean(neg_sw_mins))

# %%
print(neg_sw_windows[10] == neg_sw_windows[11])
plt.plot(neg_sw_windows[61])
plt.axvline(1000)
plt.axhline(0)
plt.show()

# %% RUN ALGO ON ALL PARTICIPANTS AND CHANNELS

DATA_DIR = Path("data/annotated")

pat = re.compile(r"^Patient(?P<p>\d+)_Channel(?P<c>\d+)_(?P<kind>.+)\.npy$")

def parse_name(fname: str):
    m = pat.match(fname)
    if not m:
        return None
    return int(m["p"]), int(m["c"]), m["kind"]

# 1) Index all files by (patient, channel, kind)
index = {}
for fp in DATA_DIR.glob("*.npy"):
    parsed = parse_name(fp.name)
    if not parsed:
        continue
    p, c, kind = parsed
    index[(p, c, kind)] = fp

# 2) Collect all EEG pairs we can run
pairs = []
for (p, c, kind), eeg_fp in index.items():
    if kind != "EEG":
        continue
    negsw_fp = index.get((p, c, "negSWs"))  # only negative slow waves
    pairs.append((p, c, eeg_fp, negsw_fp))

pairs.sort()

print(f"Found {len(pairs)} EEG channels total.")
print(f"Found {sum(1 for *_, sw in pairs if sw is not None)} with negSW annotations.")

print(pairs[3])

# %%
results = []

for p, c, eeg_fp, negsw_fp in pairs:

    ds = load_data_as_dataset(npy_path=eeg_fp, fs=sampling_rate)
    
    sos = butter(4, [0.5, 4.0], btype="bandpass", fs=ds.fs, output="sos")
    ds_filtered = sosfiltfilt(sos, ds.signal.squeeze().astype(float))

    ds_sim = Simulations.SimulationDataset(t=np.arange(len(ds_filtered)) / ds.fs,
                                           signal=ds_filtered,
                                           fs=ds.fs,
                                           name=ds.name + f"_p{p}-c{c}_ds{ds.fs}")

    result = Simulations.run_simulations(ds_sim, ZeroCross(fs=ds_sim.fs, amp_q=0.05))
    results.append(result)
    print(f"{len(results)} out of {len(pairs)}")

results = np.array(results)

# %%

print(results)

with open("results/results_zerocross_allpatients_02.pkl", "wb") as f:
    pickle.dump(results, f)
# %%
