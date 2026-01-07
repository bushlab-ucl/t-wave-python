# %% IMPORTS

import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import Simulations

# %% MAIN SCRIPT

with open("results_twave_downsampled_128.pkl", "rb") as f:
    results_twave = pickle.load(f)

results_twave.plot_timeseries(ground_truth="/home/jhedemann/slow-wave/1024hz/Patient04_OfflineMrk.mrk")
plt.show()


#print(results_twave.Dataset.t.shape)
print(len(results_twave.stims_sp))
print(np.diff(results_twave.Dataset.t)*results_twave.Dataset.fs)

print(results_twave.Dataset.t.max())

# %%
# from Simulations import PhaseTrackerStatus
# import numpy as np

# status = np.array(results_twave.status_ts)

# def count(flag):
#     return np.sum((status & flag) > 0)

# print("STIM1:", count(PhaseTrackerStatus.STIM1))
# print("INHIBITED_AMP:", count(PhaseTrackerStatus.INHIBITED_AMP))
# print("INHIBITED_RATIO:", count(PhaseTrackerStatus.INHIBITED_RATIO))
# print("INHIBITED_QUADRATURE:", count(PhaseTrackerStatus.INHIBITED_QUADRATURE))


# %%
