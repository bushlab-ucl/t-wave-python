# %% IMPORTS

from pathlib import Path
import numpy as np

from load_intracranial_data import load_data_as_dataset

import Simulations
from Algo_PLL import PhaseTracker as PLL
from Algo_TWave import PhaseTracker as TWave
from Algo_AmpTh import PhaseTracker as AmpTh
from Algo_SineFit import PhaseTracker as SineFit
from Algo_ZeroCrossing import PhaseTracker as ZeroCross

from Inhibitors import *
from tqdm import tqdm

import os
from concurrent.futures import ProcessPoolExecutor

# %% MAIN SCRIPT

ds = load_data_as_dataset(npy_path="/home/jhedemann/slow-wave/1024hz/Patient04_electrode01.npy",
                          fs=1024)



algo_names = ['PLL', 'TWave', 'AmpTh', 'SineFit', 'ZeroCrossing']
algos = {}

algos['PLL'] = PLL(
    fs=ds.fs,
    inhibitors=[MinAmp(window_length_sp=int(ds.fs * 2),
                       min_amp_threshold_uv=45)])

algos['TWave'] = TWave(ds.fs)

algos['AmpTh'] = AmpTh(stim_delay_sp=int(ds.fs * 0.5),
                       adaptive_window_sp=int(ds.fs * 5),
                       backoff_sp=int(ds.fs * 5))

algos['SineFit'] = SineFit(fs=ds.fs)
algos['ZeroCrossing'] = ZeroCross()


# results = {}
# for name, algo in algos.items():
#     results[name] = Simulations.run_simulations(ds, algo)

# results_twave = Simulations.run_simulations(ds, algos['TWave'])

ds_short = load_data_as_dataset(npy_path="/home/jhedemann/slow-wave/1024hz/Patient04_electrode01.npy",
                                fs=1024,
                                max_duration_s=300)
rslt = Simulations.run_simulations(ds_short, TWave(ds_short.fs))


# %%

plot = rslt.plot_timeseries()
plot.show()
# %%

print(rslt.stims_sp)

# %%
