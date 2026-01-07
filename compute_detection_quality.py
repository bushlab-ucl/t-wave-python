# %% IMPORTS

import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import Simulations

# %% MAIN SCRIPT

# Parse simulation results file
with open("results_twave_downsampled.pkl", "rb") as f:
    results_twave = pickle.load(f)

# Parse ground-truth markdown file
with open("/home/jhedemann/slow-wave/1024hz/Patient04_OfflineMrk.mrk", "r", encoding="utf-8") as input_file:
    text = input_file.read()

lines = text.splitlines()[1:]
ground_truth_sw_times = np.array([int(line.split()[0]) / results_twave.Dataset.fs for line in lines if line.strip()])
ground_truth_sw_times_trunc = np.array([x for x in ground_truth_sw_times if x <= results_twave.Dataset.t.max()])

print(results_twave.stims_sp / 1024)
print(ground_truth_sw_times_trunc)

# %%
