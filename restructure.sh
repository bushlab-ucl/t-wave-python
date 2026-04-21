#!/usr/bin/env bash
# Restructure the flat repo into upstream vs. Johannes's code.
#
# Run from the repo root:
#     bash restructure.sh
#
# Safe to re-run: uses `git mv` when possible, falls back to `mv`.
# After running, read RESTRUCTURE_NOTES.md for the import changes needed.

set -e

USE_GIT=0
if [ -d .git ] && command -v git >/dev/null 2>&1; then
    USE_GIT=1
fi

move() {
    src="$1"
    dst="$2"
    if [ ! -e "$src" ]; then
        echo "  skip (not found): $src"
        return
    fi
    mkdir -p "$(dirname "$dst")"
    if [ "$USE_GIT" = "1" ]; then
        git mv "$src" "$dst"
    else
        mv "$src" "$dst"
    fi
    echo "  $src  →  $dst"
}

echo "Creating directory structure..."
mkdir -p t-wave-algo
mkdir -p johannes-code
mkdir -p johannes-code/validation

echo ""
echo "Moving upstream t-wave-algo files..."
move "Algo_PLL.py"                 "t-wave-algo/Algo_PLL.py"
move "Algo_TWave.py"               "t-wave-algo/Algo_TWave.py"
move "Algo_AmpTh.py"               "t-wave-algo/Algo_AmpTh.py"
move "Algo_SineFit.py"             "t-wave-algo/Algo_SineFit.py"
move "Algo_ZeroCrossing.py"        "t-wave-algo/Algo_ZeroCrossing.py"
move "Simulations.py"              "t-wave-algo/Simulations.py"
move "Inhibitors.py"               "t-wave-algo/Inhibitors.py"
move "run_group_simulations.ipynb" "t-wave-algo/run_group_simulations.ipynb"
move "README.md"                   "t-wave-algo/README.md"

echo ""
echo "Moving Johannes's code..."
move "load_intracranial_data.py"      "johannes-code/load_intracranial_data.py"
move "run_algos.py"                   "johannes-code/run_algos.py"
move "run_twave.py"                   "johannes-code/run_twave.py"
move "run_zero_crossing.py"           "johannes-code/run_zero_crossing.py"
move "run_ampthreshold.py"            "johannes-code/run_ampthreshold.py"
move "load_and_test_results.py"       "johannes-code/load_and_test_results.py"
move "diagnostics.py"                 "johannes-code/diagnostics.py"
move "compute_detection_quality.py"   "johannes-code/compute_detection_quality.py"
move "analyze_time_frequency.py"      "johannes-code/analyze_time_frequency.py"
move "test_loadsleepedf.ipynb"        "johannes-code/test_loadsleepedf.ipynb"
# note: filename has a typo (comma instead of dot) — fix on move
move "troubleshoot_zero_crossing,py"  "johannes-code/troubleshoot_zero_crossing.py"

# ns6/npz data-prep lives under johannes-code/validation
move "validation/ns6_to_npz.py"       "johannes-code/validation/ns6_to_npz.py"
# if the validation/ dir became empty, remove it
rmdir validation 2>/dev/null || true

echo ""
echo "Done. Files moved."
echo ""
echo "Next steps:"
echo "  1. Read RESTRUCTURE_NOTES.md — imports in johannes-code/*.py need updating."
echo "  2. Keep pyproject.toml / requirements.txt / .gitignore at the repo root."
echo "  3. Run your scripts from inside johannes-code/ (not the repo root)."
