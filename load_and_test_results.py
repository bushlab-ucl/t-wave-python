# %% IMPORTS

import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from Simulations import PhaseTrackerStatus


# %% LOAD DATA

with open("results/results_twave_patient03_channel1_22_param_changed.pkl", "rb") as f:
    results_twave = pickle.load(f)

# %% SHOW MASTER PLOT OF RESULTS WITH GROUND TRUTH ANNOTATION

results_twave.plot_timeseries(ground_truth_sw="/home/jhedemann/slow-wave/annotated/Patient03_Channel1_SWs.npy",
                              ground_truth_ied="/home/jhedemann/slow-wave/annotated/Patient03_Channel1_IEDs.npy")
plt.show()


#print(results_twave.Dataset.t.shape)
print(len(results_twave.stims_sp))
print(np.diff(results_twave.Dataset.t)*results_twave.Dataset.fs)

print(results_twave.Dataset.t.max())
# %%
