# Self-Similarity (SS) Algorithm

This directory contains the necessary scripts for running the self-similarity (SS) algorithm on respiratory signal data:

- **`compute_SS.py`**: Primary script that ties everything together to run the SS algorithm, including reading input files, invoking `run_SS_detector.py`, and saving the resulting output.
- **`SS_detector/run_SS_detector.py`**: Functions that orchestrate the SS detection logic.
- **`SS_detector/Algorithm_functions.py`**: Core utility functions used for data preprocessing, loading, and filtering.

## Overview

The SS algorithm identifies and quantifies high loop gain (HLG) breathing oscillations — referred to as “self-similar” patterns — in respiratory signals. The scripts here:

1. **Load Prepared Data**: Retrieves or prepares respiration data, often from HDF5 files.
2. **Preprocessing & Flow Limitations**: Applies a series of filtering and threshold-based steps to identify apnea, hypopnea, and other respiratory events.
3. **Self-Similarity Detection**: Identifies repeating breathing complexes that meet duration, shape, and amplitude criteria.
4. **Output Generation**: Stores results (e.g., SS arrays, summary metrics) back to HDF5 files or CSV reports.

## Requirements

Make sure you have Python 3 installed along with these packages:

```bash
pip install numpy pandas matplotlib scipy seaborn scikit-learn h5py statsmodels
```

## Usage

### 1. Prepare Your Data
- Ensure your input data (e.g., preprocessed sleep recordings in HDF5 files) is accessible by the `compute_SS.py` script.

### 2. Configure Your Script (Optional)
- In **`SS_detector/Algorithm_functions.py`**, you can adjust thresholds for filtering out noisy signals, or define how to treat events.

### 3. Run the SS Algorithm
- From a terminal, navigate to this directory and execute:

```bash
python compute_SS.py
```

- The script will:
  1. Locate the input HDF5 files.
  2. Load and preprocess each file’s respiratory data.
  3. Call **`run_SS_detector.py`** to compute self-similarity events.
  4. Save the output (e.g., SS-based arrays and indexes) back into the same or new HDF5 files.

### 4. Review Outputs
- Check the output HDF5 files or any CSV logs to confirm the presence of SS-related data (like `E_sim`, `T_sim`, or `TAGGED` columns). These columns indicate the presence of self-similar breathing oscillations.

## Customizing

- Modify threshold parameters for event detection in `SS_detector/Algorithm_functions.py` if your data characteristics differ from the defaults.
- Tune the self-similarity acceptance criteria in `SS_detector/run_SS_detector.py` for different definitions of HLG breathing.



