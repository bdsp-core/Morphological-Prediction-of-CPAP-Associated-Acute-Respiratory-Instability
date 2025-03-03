# SS Algorithm Standalone

A standalone Python script for advanced respiratory signal processing and event detection. This repository contains code for processing EDF files, extracting respiratory channels, applying preprocessing steps, and detecting flow reductions (apneas, hypopneas) and self-similarity events in respiratory signals.

## Overview

The SS Algorithm processes:
- **List and load EDF files:** Automatically scans a folder for EDF files.
- **Channel detection:** Searches for respiratory channels using flexible string matching.
- **Preprocessing:** Includes NaN removal, signal clipping, and normalization.
- **Envelope & Baseline Computation:** Computes positive and negative envelopes along with a dynamic baseline.
- **Event Detection:** Implements methods to detect respiratory events (apneas, hypopneas) based on thresholding and window correction.
- **Self-Similarity Analysis:** Identifies potential self-similarity regions in the respiratory signals.
- **Reporting & Visualization:** Provides summary reports and plotting functions for a detailed analysis of respiratory events.

## Features

- **Flexible Channel Search:** Automatically identifies key respiratory channels (e.g., abdomen, chest, airflow).
- **Signal Preprocessing:** Functions to remove artifacts, outliers, and handle missing values.
- **Dynamic Thresholding:** Uses rolling quantiles and window corrections to compute dynamic excursion thresholds.
- **Event Connectivity:** Connects short events to form clinically relevant patterns.
- **Visualization:** Includes plotting routines to visualize respiratory signals, detected events, and self-similarity metrics.
- **Modular Design:** Well-organized functions that can be individually integrated or modified for specific research needs.

## Installation

Ensure you have Python 3 installed. Then, install the required dependencies:

```bash
pip install numpy pandas matplotlib scipy mne
```


## Usage

1. **Prepare your data:**
   - Place your EDF files in a dedicated folder.
2. **Run the script:**
   - Execute the script in the terminal:
     ```bash
     python SS_algorithm_standalone.py
     ```
   - The script will list available EDF files, process the selected file(s), and save results in a directory named `Results`.
3. **Review the output:**
   - A summary report and visualizations will be generated, providing details on detected events and overall signal quality.

## Code Structure

- **list_files:** Searches for EDF files in a given folder.
- **multiple_channel_search & original_effort_search:** Functions to locate respiratory channels.
- **Preprocessing Functions:** Includes functions such as `remove_nans`, `cut_flat_signal`, and `do_initial_preprocessing`.
- **Envelope & Event Detection:** Functions like `compute_envelopes`, `find_events`, and `label_correction` perform the core analysis.
- **Reporting & Visualization:** Generates summary reports and plots the processed signal data.

## Credits & Ownership

> **Notice: Ownership and Patent Information**  
> This program, including its underlying algorithms and methodologies, is owned and patented by: T. Nassi, E. Oppersma, M.B. Westover, and R.J. Thomas.  
> Any unauthorized use, reproduction, or distribution of this program or its components is strictly prohibited and may result in legal action.  
> For licensing information, please contact the corresponding author (R.J. Thomas).

## License

_Include your license information here (if applicable)._

## Contributing

Contributions are welcome. Please fork the repository and submit a pull request with your changes. For major changes, open an issue first to discuss what you would like to change.

## Contact

For questions, suggestions, or licensing inquiries, please contact:
- **T. Nassi**
- **E. Oppersma**
- **M. B. Westover**
- **R.J. Thomas**

