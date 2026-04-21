# TWave slow-wave detection — offline pipeline

Runs the TWave phase-tracker algorithm on an EEG recording, produces a detection log, a summary figure, and a per-event plot for every detected stim.

This is the version to play with on existing data while I'm away. Full validation and comparison against the old zero-crossing approach is deferred to June — see "Status" at the bottom.

## Repo layout

```
.
├── t-wave-algo/                 # upstream TWave code (Simulations, Algo_*, Inhibitors)
├── johannes-code/               # Johannes's intracranial-data pipeline & analyses
│   └── validation/
│       └── run_twave.ipynb      # ← the notebook to run
├── data/
│   └── annotated/               # example .npy files with ground-truth SW/IED labels
├── output/                      # generated logs and figures land here
├── pyproject.toml
└── README.md
```

## How to run

1. Open `johannes-code/validation/run_twave.ipynb` in Jupyter or VS Code.
2. In the **CONFIG** cell, set:
   - `input_path` — path to your EEG file (`.npy` or `.npz`)
   - `fs` — sample rate in Hz (ignored for `.npz`, which carries its own fs)
   - `channel_idx` — which channel to use (if multi-channel)
   - `amp_threshold_uv` / `amp_limit_uv` — amplitude gates (see below)
3. Run all cells top to bottom.

## What you get

Three outputs land in `output/`:

- **`<filename>_stims.csv`** — one row per detected stim with sample index, time, and TWave's estimated frequency/amplitude/phase at that point.
- **`<filename>_summary.png`** — 3-panel figure: stim-triggered average SO-band waveform, polar histogram of stim phases, top-6 tracker-status counts.
- **`<filename>_result.pkl`** — the full `SimulationResult` object (for downstream analysis notebooks).

Plus inline plots in the notebook: one figure per detected stim showing ±4 s of context with raw signal (grey), SO-band filtered (blue), and the stim marker (red).

## Reading the output

**Status breakdown** (printed before the summary figure) is the first thing to look at. Each sample of the recording gets a status flag; the ones you care about:

- `STIM1` — a stim fired. In a good run you want hundreds or thousands of these over a full-night recording.
- `INHIBITED_AMP` — signal amplitude outside the `[amp_threshold_uv, amp_limit_uv]` gate. If this dominates, your amp gates are wrong for this dataset.
- `INHIBITED_QUADRATURE` — signal didn't match a sinusoidal template well enough.
- `INHIBITED_RATIO` — too much high-frequency content relative to low (IED rejection).
- `WRONGPHASE` — tracker was healthy but we weren't at a target phase yet.
- `BACKOFF` / `BACKOFF_ISI` — refractory periods after a stim.

**Summary figure:**
- Panel (a) should show a clean deflection at `t=0` if stims are genuinely locking onto slow waves. Flat = firing on noise.
- Panel (b) should have red (actual mean phase) and green (target) close together. Error in degrees printed in the title.
- Panel (c) tells you whether the tracker was mostly firing, mostly blocked, or stuck in backoff.

## Tuning

TWave has ~8 interacting parameters, all with defaults tuned to the specific intracranial data Johannes was working with. **Defaults do not transfer cleanly to new datasets.** Signs you need to re-tune:

| Symptom | Parameter | Where |
| --- | --- | --- |
| Almost all samples are `INHIBITED_AMP`, very few stims | `amp_threshold_uv`, `amp_limit_uv` | CONFIG cell (already exposed) |
| `INHIBITED_QUADRATURE` dominates | `quadrature_thresh` (default 0.5) | `t-wave-algo/Algo_TWave.py`, or set `tw.quadrature_thresh = …` |
| `INHIBITED_RATIO` dominates | `high_low_freq_ratio` (default 0.50) | same pattern |
| Summary panel (b) shows large phase error | `prediction_limit_s` (default 0.15) | same pattern |

The CONFIG cell prints your signal's amplitude percentiles before running — use those to pick `amp_threshold_uv` (≈ 50–75th percentile) and `amp_limit_uv` (≈ 99th × 1.5).

There's a known bug in upstream TWave's kwargs override (prints with `:s` format on int values and crashes). The notebook works around it by setting attributes after construction:

```python
tw = TWave(fs)
tw.amp_threshold_uv = 75
tw.amp_limit_uv = 3000
```

Use the same pattern for any other parameter you want to override.

## File formats supported

**`.npy`** — raw 1-D array (µV) or 2-D (n_samples, n_channels). You supply `fs` via the config.

**`.npz`** — format produced by `johannes-code/validation/ns6_to_npz.py` from Blackrock `.ns6` files. Must contain:
- `data`: `(n_samples, n_channels)` int16 or float32
- `fs`: sample rate (scalar)
- `scale_factors`: `(n_channels,)` — multiply int16 values to get µV (optional if data is already float32 µV)

## Status — what works, what doesn't

**Works:**
- Pipeline runs end-to-end on existing `.npy` files from `data/annotated/`.
- Pipeline runs end-to-end on `.npz` files converted from Blackrock `.ns6`.
- Output formats (CSV + summary PNG + per-event plots + pickled result) are all generated.

**Doesn't work yet:**
- **Parameter tuning is manual per-dataset.** TWave is parameter-sensitive and the defaults don't transfer. The NS6 test file I converted (`20190122-065346-004.npz`) produces poor output with defaults; likely needs re-tuned amp/quadrature/ratio thresholds.
- **No automated ground-truth validation.** Johannes wrote a precision/sensitivity scorer (`johannes-code/compute_detection_quality.py`) that works against the annotated `.npy` files, but Johannes and I both have concerns about those labels — worth revisiting before treating F1 numbers as authoritative.
- **No side-by-side comparison with the old zero-crossing approach.** The old Rust-based pipeline is a separate repo.

## For June

When I'm back in London (full month, June):

1. Set up both TWave and the old zero-crossing/trough detector to run on the same recordings.
2. Get Laurent to validate a labeled subset we both trust.
3. Compare detector performance on that subset (sensitivity, specificity, phase accuracy).
4. Pick whichever is more robust + auditable for real-data use.
5. Revisit the Blackrock NPlay/Cerberus live pipeline once offline validation is solid.

## Dependencies

Python ≥ 3.12. Install via `pip install -r requirements.txt` or the `pyproject.toml`. Main deps: numpy, scipy, matplotlib.

## Who wrote what

- `t-wave-algo/` — original authors of the TWave paper / reference implementation. Not modified in this fork (parameter tweaks are applied at runtime in the notebook, not baked into the upstream code).
- `johannes-code/` — Johannes Hedemann's intracranial-data pipeline, diagnostics, and analyses.
- `johannes-code/validation/` — my additions: ns6→npz converter and the `run_twave.ipynb` notebook.
