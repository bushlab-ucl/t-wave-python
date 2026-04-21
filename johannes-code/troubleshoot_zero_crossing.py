# %% IMPORTS

import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import Simulations
from Simulations import PhaseTrackerStatus

# %% LOAD DATA

patient03_channel1_eeg = np.load("data/annotated/Patient03_Channel1_EEG.npy")
patient03_channel1_sws = np.load("data/annotated/Patient03_Channel1_negSWs.npy")
patient03_channel1_ieds = np.load("data/annotated/Patient03_Channel1_IEDs.npy")

with open("results/results_zerocross_patient03_channel1_08_newbackoff_sp.pkl", "rb") as f:
    results_twave = pickle.load(f)

# %% 

# how close was the algorithm to firing at ground truth negative slow waves?

# 