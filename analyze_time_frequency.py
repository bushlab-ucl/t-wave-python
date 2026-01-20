# %% IMPORTS

import numpy as np
import matplotlib.pyplot as plt
import mne
from pathlib import Path

from load_intracranial_data import load_data_as_dataset

# %% CONFIG

time_excerpt = 0 # seconds
sampling_rate = 512 # hz
ds = load_data_as_dataset(npy_path="/home/jhedemann/ptas_benchmarks_jhedemann/data/annotated/Patient03_Channel1_EEG.npy",
                          fs=sampling_rate)
if time_excerpt != 0:
    ds_trunc = ds.signal.squeeze().astype(float)[:time_excerpt*sampling_rate]
else:
    ds_trunc = ds.signal.squeeze().astype(float)

freq_range = np.linspace(start=1, stop=200, num=50)

ds_tfr_reshape = np.reshape(ds_trunc, (1, 1, ds_trunc.shape[0]))

ground_truth_sw="/home/jhedemann/ptas_benchmarks_jhedemann/data/annotated/Patient03_Channel1_SWs.npy"
ground_truth_ied="/home/jhedemann/ptas_benchmarks_jhedemann/data/annotated/Patient03_Channel1_IEDs.npy"

path_sw = Path(ground_truth_sw)
path_ied = Path(ground_truth_ied)

arr_sw = np.load(path_sw)
arr_ied = np.load(path_ied)

if time_excerpt != 0:
    arr_sw_trunc = np.array([x for x in arr_sw if x <= time_excerpt])
    arr_ied_trunc = np.array([x for x in arr_ied if x <= time_excerpt])
else:
    arr_sw_trunc = arr_sw
    arr_ied_trunc = arr_ied


# %%
print((arr_ied_trunc*512).astype(int))

# %% FUNCTIONS

def filter_for_competing_events(events_1, events_2, buffer_s):

    helper_1 = events_1
    helper_2 = events_2

    events_1 = [i for i in events_1 if
                    np.min(np.abs(helper_2 - i)) > buffer_s]
    events_2 = [i for i in events_2 if
                    np.min(np.abs(helper_1 - i)) > buffer_s]

    return np.array(events_1), np.array(events_2)

def find_maxima(signal, times, fs, window_size_s, low_pass=None):
    max_idx = []
    
    # low-pass filter signal such that maximum is not high-freq jitter
    if low_pass is not None:
        search_signal = mne.filter.filter_data(
            signal.astype(float), fs, l_freq=None, h_freq=low_pass, verbose=False
        )
    else:
        search_signal = signal

    idx = (times * fs).astype(int).ravel()
    window_size_idx = int(window_size_s * fs)
    sig_len = len(signal)

    for i in idx:

        # lookout for signal boundaries
        start = max(0, i - window_size_idx)
        end = min(sig_len, i + window_size_idx)
        
        window = search_signal[start:end]
        
        if len(window) == 0:
            continue
            
        max_i = np.argmax(window) + start
        max_idx.append(max_i)
    
    return np.array(max_idx)

def get_signal_subsets_from_events(signal, idx, fs, window_size_s):
    windows = []
    window_size_idx = int(window_size_s * fs)
    expected_length = 2 * window_size_idx
    sig_len = len(signal)

    for i in idx:
        i = int(i)
        # Define the theoretical boundaries
        start = i - window_size_idx
        end = i + window_size_idx
        
        # Define the actual boundaries (clamped to signal limits)
        actual_start = max(0, start)
        actual_end = min(sig_len, end)
        
        # Slice the available signal
        chunk = signal[actual_start:actual_end]
        
        if len(chunk) == expected_length:
            windows.append(chunk)
        else:
            # CREATE THE PADDING
            # 1. Start with an array of zeros of the exact target shape
            padded_window = np.zeros(expected_length)
            
            # 2. Calculate where to place the 'chunk' within the padded array
            # If start was -10, the chunk should start at index 10 of the padded array
            pad_start = actual_start - start 
            pad_end = pad_start + len(chunk)
            
            padded_window[pad_start:pad_end] = chunk
            windows.append(padded_window)
    
    return np.array(windows)

def find_empty_windows(sw_times, ied_times, fs, windows_size_s, empty_space_s, n_windows):

    # get all events into a sorted array
    total_times = np.concatenate([sw_times.ravel(), ied_times.ravel()])
    total_times = (total_times*fs).astype(int)
    total_times = np.sort(total_times)

    # keep appending intervals until next event is too close

    helper = np.insert(total_times, 0, 0)
    empty_windows = []
    len_interval = windows_size_s * fs
    safety_buffer = (empty_space_s - windows_size_s) / 2

    for i in range(len(helper)-1):

        safe_i = helper[i] + safety_buffer

        while (safe_i + len_interval + safety_buffer) < helper[i+1]:

            this_interval = (safe_i, safe_i+len_interval)
            empty_windows.append(this_interval)
            safe_i += len_interval

    return np.array(empty_windows).astype(int)

def get_signal_subsets_from_intervals(signal, intervals):

    signal_subsets = []

    for interval in intervals:

        start = interval[0]
        end = interval[1]

        this_subset = signal[start:end]
        signal_subsets.append(this_subset)
        
    return np.array(signal_subsets)

if __name__ == "__main__":
    # %% PREPARE DATA

    filtered_arr_ied, filtered_arr_sws = filter_for_competing_events(arr_ied_trunc, arr_sw_trunc, buffer_s=4)

    ied_maxima = find_maxima(ds_trunc, filtered_arr_ied, sampling_rate, 4, low_pass=4)
    ied_subsets = get_signal_subsets_from_events(ds_trunc, ied_maxima, sampling_rate, 4)
    ied_subsets = np.expand_dims(ied_subsets, axis=1)

    sw_maxima = find_maxima(ds_trunc, filtered_arr_sws, sampling_rate, 4, low_pass=4)
    sw_subsets = get_signal_subsets_from_events(ds_trunc, sw_maxima, sampling_rate, 4)
    sw_subsets = np.expand_dims(sw_subsets, axis=1)

    total_intervals = find_empty_windows(sw_times=arr_sw_trunc,
                                ied_times=arr_ied_trunc,
                                fs=sampling_rate,
                                windows_size_s=8,
                                empty_space_s=12,
                                n_windows=max((len(arr_sw_trunc), len(arr_ied_trunc))))

    signal_without_events = get_signal_subsets_from_intervals(signal=ds_trunc,
                                                            intervals=total_intervals)
    signal_without_events = np.expand_dims(signal_without_events, axis=1)

    # %% TIME FREQUENCY ANALYSIS

    tfr_signal_total = mne.time_frequency.tfr_array_morlet(data=ds_tfr_reshape,
                                                        sfreq=sampling_rate,
                                                        freqs=freq_range,
                                                        n_cycles=5,
                                                        verbose=True)

    tf_total = tfr_signal_total[0,0,:,:]
    power_total = np.abs(tf_total) ** 2

    tfr_signal_no_events = mne.time_frequency.tfr_array_morlet(data=signal_without_events,
                                                            sfreq=sampling_rate,
                                                            freqs=freq_range,
                                                            n_cycles=5,
                                                            verbose=True)

    tf_no_events = tfr_signal_no_events[:,0,:,:]
    power_no_events = np.abs(tf_no_events) ** 2
    avg_power_no_events = np.average(power_no_events, axis=(0,2))
    avg_power_no_events = avg_power_no_events[:, np.newaxis]

    tfr_signal_ieds = mne.time_frequency.tfr_array_morlet(data=ied_subsets,
                                                        sfreq=sampling_rate,
                                                        freqs=freq_range,
                                                        n_cycles=5,
                                                        verbose=True)

    tf_ieds = tfr_signal_ieds[:,0,:,:]
    power_ieds = np.abs(tf_ieds) ** 2
    avg_power_ieds = np.average(power_ieds, axis=0)

    tfr_signal_sws = mne.time_frequency.tfr_array_morlet(data=sw_subsets,
                                                        sfreq=sampling_rate,
                                                        freqs=freq_range,
                                                        n_cycles=5,
                                                        verbose=True)

    tf_sws = tfr_signal_sws[:,0,:,:]
    power_sws = np.abs(tf_sws) ** 2
    avg_power_sws = np.average(power_sws, axis=0)

    base_correct_ieds = np.log(avg_power_ieds) - np.log(avg_power_no_events)
    base_correct_sws = np.log(avg_power_sws) - np.log(avg_power_no_events)

    # %% PLOTS
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))


    cf = axs[0].contourf(
        np.linspace(0, time_excerpt, power_total.shape[1]),  # time axis
        freq_range,                                     # frequency axis
        np.log(power_total),                                          # (freq x time)
        levels=50,
        cmap="viridis",
    )

    [axs[0].axvline(sw, 0.25, 0.5, color="white", alpha=0.2) for sw in arr_sw_trunc]
    [axs[0].axvline(ied, 0.25, 0.5, color="red", alpha=1) for ied in arr_ied_trunc]
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Frequency (Hz)")

    fig.colorbar(cf, ax=axs[0], label="Power")

    axs[1].psd(x=ds_trunc, Fs=sampling_rate)

    plt.show()

    # %%

    # create global value range
    all_values = np.concatenate([base_correct_ieds.ravel(), base_correct_sws.ravel()])
    min_val = np.min(all_values)
    max_val = np.max(all_values)

    # center value range around 0
    limit = max(abs(min_val), abs(max_val))
    vmin, vmax = -limit, limit

    levels = np.linspace(vmin, vmax, 50)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    ied_cf = axs[0].contourf(np.linspace(-4, 4, base_correct_ieds.shape[1]), # time axis
                        freq_range, # frequency axis
                        base_correct_ieds, # (freq x time)
                        levels=levels,
                        vmin=vmin, vmax=vmax,
                        cmap="viridis")

    sw_cf = axs[1].contourf(np.linspace(-4, 4, base_correct_sws.shape[1]), # time axis
                        freq_range, # frequency axis
                        base_correct_sws, # (freq x time)
                        levels=levels,
                        vmin=vmin, vmax=vmax,
                        cmap="viridis")

    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Frequency (Hz)")
    axs[0].set_title("IEDs")

    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Frequency (Hz)")
    axs[1].set_title("SWs")

    # fig.colorbar(ied_cf, ax=axs[0], label="Power")
    # fig.colorbar(sw_cf, ax=axs[1], label="Power")

    fig.colorbar(sw_cf, ax=axs, label="Log Power Ratio (Event / Baseline)")

    plt.show()
