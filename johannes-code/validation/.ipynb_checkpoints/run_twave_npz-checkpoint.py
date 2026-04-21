# %% IMPORTS

import sys
from pathlib import Path

# reach into ../../t-wave-algo/ for upstream modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "t-wave-algo"))

import numpy as np
import pickle
from scipy.signal import butter, sosfiltfilt, resample_poly

from Simulations import SimulationDataset, run_simulations
from Algo_TWave import PhaseTracker as TWave


# %% CONFIG

npz_path = "/data/20190122-065346-004.npz"
out_path = "results/results_twave_20190122-065346-004.pkl"

channel_idx = 0          # which channel to run on
target_fs = 512          # Hz — downsample to this for TWave
time_excerpt_s = 0       # 0 = use whole recording; otherwise seconds from start
bandpass_hz = None       # e.g. (0.3, 4.0) to bandpass before running; None to skip


# %% LOAD NPZ (matches the format produced by ns6_to_npz.py)

def load_ns6_npz(path, channel_idx=0, max_duration_s=None):
    path = Path(path)
    npz = np.load(path, allow_pickle=False)

    print(f"Arrays in npz: {list(npz.files)}")

    data = npz["data"]                         # (n_samples, n_channels)
    fs_native = float(npz["fs"])               # scalar
    scale_factors = npz["scale_factors"]       # (n_channels,)
    labels = npz["labels"] if "labels" in npz.files else None
    units = npz["units"] if "units" in npz.files else None

    print(f"  data shape : {data.shape}  dtype={data.dtype}")
    print(f"  fs_native  : {fs_native} Hz")
    print(f"  duration   : {data.shape[0] / fs_native:.1f} s")
    if labels is not None:
        print(f"  labels[:5] : {labels[:5]}")
    if units is not None:
        print(f"  units[0]   : {units[0]}")

    if channel_idx >= data.shape[1]:
        raise IndexError(f"channel_idx={channel_idx} but only {data.shape[1]} channels")

    sig = data[:, channel_idx]

    # Convert to µV if still raw int16. If already float32, assume ns6_to_npz
    # was called with --uv and data is already in µV.
    if np.issubdtype(sig.dtype, np.integer):
        sig = sig.astype(np.float64) * float(scale_factors[channel_idx])
        print(f"  scaled int16 → µV using factor {scale_factors[channel_idx]:.6g}")
    else:
        sig = sig.astype(np.float64)
        print("  data already float; assuming µV")

    if max_duration_s is not None and max_duration_s > 0:
        sig = sig[: int(max_duration_s * fs_native)]

    return sig, fs_native


sig_native, fs_native = load_ns6_npz(
    npz_path,
    channel_idx=channel_idx,
    max_duration_s=time_excerpt_s if time_excerpt_s else None,
)


# %% DOWNSAMPLE TO target_fs
# NS6 is typically 30 kHz — way too high for TWave (wavelet length scales with fs).
# resample_poly includes an anti-alias filter.

if abs(fs_native - target_fs) < 1e-6:
    sig = sig_native
    fs = fs_native
    print(f"No resampling needed (fs={fs} Hz)")
else:
    from math import gcd
    g = gcd(int(round(fs_native)), int(target_fs))
    up = int(target_fs) // g
    down = int(round(fs_native)) // g
    print(f"Resampling {fs_native} Hz → {target_fs} Hz   (up={up}, down={down})")
    sig = resample_poly(sig_native, up=up, down=down)
    fs = target_fs


# %% OPTIONAL BANDPASS

if bandpass_hz is not None:
    sos = butter(4, bandpass_hz, btype="bandpass", fs=fs, output="sos")
    sig = sosfiltfilt(sos, sig)
    print(f"Bandpassed {bandpass_hz[0]}–{bandpass_hz[1]} Hz")


# %% BUILD DATASET AND RUN TWAVE

sig = np.asarray(sig, dtype=float).squeeze()
print(f"\nFinal signal: {sig.shape[0]} samples, {sig.shape[0] / fs:.1f} s at {fs} Hz")

ds_sim = SimulationDataset(
    t=np.arange(len(sig)) / fs,
    signal=sig,
    fs=fs,
    name=Path(npz_path).stem + f"_ch{channel_idx}_fs{fs}",
)

rslt = run_simulations(ds_sim, TWave(ds_sim.fs))


# %% SAVE

out_path = Path(out_path)
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "wb") as f:
    pickle.dump(rslt, f)

print(f"\nSaved → {out_path}")
print(f"Number of stims: {len(rslt.stims_sp)}")
if len(rslt.stims_sp) > 0:
    print(f"First stim times (s): {[round(s/fs, 3) for s in rslt.stims_sp[:10]]}")
