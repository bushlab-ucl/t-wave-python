# %% IMPORTS

import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from load_intracranial_data import load_sw_annotation

from Simulations import PhaseTrackerStatus

# %% LOAD DATA

with open("results/results_zerocross_patient03_channel1_19_newdynamic.pkl", "rb") as f:
    results_twave = pickle.load(f)

# %%
print(results_twave.PhaseTracker.__dict__)
# %% SHOW MASTER PLOT OF RESULTS WITH GROUND TRUTH ANNOTATION

results_twave.plot_timeseries(ground_truth_sw="data/annotated/Patient03_Channel1_negSWs.npy",
                              ground_truth_ied="data/annotated/Patient03_Channel1_IEDs.npy")
plt.show()


#print(results_twave.Dataset.t.shape)
print(len(results_twave.stims_sp))
print(np.diff(results_twave.Dataset.t)*results_twave.Dataset.fs)

print(results_twave.Dataset.t.max())

# %%

neg_sw_annot = "/home/jhedemann/ptas_benchmarks_jhedemann/data/annotated/Patient03_Channel1_negSWs.npy"
arr_neg_sw = load_sw_annotation(neg_sw_annot)
print(len(arr_neg_sw))

