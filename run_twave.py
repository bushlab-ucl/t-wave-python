# %% IMPORTS

import numpy as np
import pickle

from load_intracranial_data import load_data_as_dataset

import Simulations
from Algo_TWave import PhaseTracker as TWave


# %%

time_excerpt = 600 # seconds
sampling_rate = 512 # hz
ds = load_data_as_dataset(npy_path="/home/jhedemann/slow-wave/annotated/Patient03_Channel1_EEG.npy",
                          fs=sampling_rate)
if time_excerpt != 0:
    ds_trunc = ds.signal.squeeze().astype(float)[:time_excerpt*sampling_rate]
else:
    ds_trunc = ds.signal.squeeze().astype(float)

#%% 

print("ds.fs:", ds.fs)
print("ds.signal.shape:", np.asarray(ds.signal).shape)
print("ds_trunc.shape:", ds_trunc.shape)
print("len(ds_trunc):", len(ds_trunc))
print("duration_s (from ds_trunc):", len(ds_trunc)/ds.fs)

# %%

ds_sim = Simulations.SimulationDataset(
    t=np.arange(len(ds_trunc)) / ds.fs,
    signal=ds_trunc,
    fs=ds.fs,
    name=ds.name + f"_p03-c1_ds{ds.fs}",
)

rslt = Simulations.run_simulations(ds_sim, TWave(ds_sim.fs))

# %%

with open("results/results_twave_patient03_channel1_22_param_changed.pkl", "wb") as f:
    pickle.dump(rslt, f)

# %%

print(rslt.stims_sp)

# %%
