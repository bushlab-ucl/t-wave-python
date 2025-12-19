# %% IMPORTS

import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import Simulations

# %% MAIN SCRIPT

with open("results_twave_downsampled.pkl", "rb") as f:
    results_twave = pickle.load(f)

results_twave.plot_timeseries()
plt.show()


# %%
from Simulations import PhaseTrackerStatus
import numpy as np

status = np.array(results_twave.status_ts)

def count(flag):
    return np.sum((status & flag) > 0)

print("STIM1:", count(PhaseTrackerStatus.STIM1))
print("INHIBITED_AMP:", count(PhaseTrackerStatus.INHIBITED_AMP))
print("INHIBITED_RATIO:", count(PhaseTrackerStatus.INHIBITED_RATIO))
print("INHIBITED_QUADRATURE:", count(PhaseTrackerStatus.INHIBITED_QUADRATURE))


# %%
