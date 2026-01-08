# %% IMPORTS

import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import Simulations

# %% PREP

# Parse simulation results file
with open("results/results_twave_patient03_channel1_03.pkl", "rb") as f:
    results_twave = pickle.load(f)

# Parse ground_truth npy files
path_sw = Path("/home/jhedemann/slow-wave/annotated/Patient03_Channel1_SWs.npy")
path_ied = Path("/home/jhedemann/slow-wave/annotated/Patient03_Channel1_IEDs.npy")

arr_sw = np.load(path_sw)
arr_ied = np.load(path_ied)

arr_sw_trunc = np.array([x for x in arr_sw if x <= results_twave.Dataset.t.max()])
arr_ied_trunc = np.array([x for x in arr_ied if x <= results_twave.Dataset.t.max()])

print(len(arr_sw_trunc))

sim_stim_times = np.array(results_twave.stims_sp) / results_twave.Dataset.fs

#print(arr_sw_trunc)
#print(arr_ied_trunc)

# %% FUNCTIONS

def compute_detection_quality(detected_sim, detected_true, tol=0.1):
    detected_sim = np.array(detected_sim)
    detected_true = np.array(detected_true)

    used_true = np.zeros(len(detected_true), dtype=bool)

    TP = 0
    FP = 0

    for d in detected_sim:
        # compute distances to every unused ground truth event
        diffs = np.abs(detected_true - d)

        # find closest ground truth slow wave
        i = np.argmin(diffs)
        if diffs[i] <= tol and not used_true[i]:
            TP += 1
            used_true[i] = True
        else:
            FP += 1

    FN = int(np.sum(~used_true))

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    f1 = 2 * TP / (2 * TP + FP + FN) if (2*TP + FP + FN) > 0 else np.nan

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "sensitivity": float(sensitivity),
        "precision": float(precision),
        "f1": float(f1)
    }

# %% Main SCRIPT

tol_qualities = []
tol_range = range(10)

for tol in tol_range:
    tol_qualities.append(compute_detection_quality(sim_stim_times, arr_sw_trunc, tol=tol))

sens, prec, f1 = ([q["sensitivity"] for q in tol_qualities],
                  [q["precision"] for q in tol_qualities],
                  [q["f1"] for q in tol_qualities])

plt.figure(figsize=(8, 5))

plt.plot(tol_range, sens, label="sensitivity (true detected / all slow waves)")
plt.plot(tol_range, prec, label="precision (true detected / all detected)")
plt.plot(tol_range, f1, label="f1 score")

plt.xlabel("tolerance (s)")
plt.ylabel("score")
#plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()



# %%
