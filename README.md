# SS Algorithm Standalone

A standalone Python script for advanced respiratory signal processing and event detection. This repository contains code for processing EDF files, extracting respiratory channels, applying preprocessing steps, and detecting flow reductions (apneas, hypopneas) and self-similarity events in respiratory signals.

## Overview

The SS Algorithm processes electrophysiological data (typically EDF files) to:
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
- **R.J. Thomas**, **T. Nassi**.


# Figures and Analysis Scripts

This repository also contains Python scripts for generating various statistical and performance-related figures, particularly for CPAP success prediction, self-similarity analysis, and model evaluation.

## Overview

The scripts provided perform the following key tasks:
- **Pie Charts, Histograms & Heatmaps:** Offer visual insights into data distribution, patient stratification, and CPAP therapy outcomes.
- **ROC, PRC & Calibration Curves:** Compute and plot Receiver Operating Characteristic (ROC) curves, Precision-Recall (PR) curves, and calibration plots to assess the performance of predictive models.
- **Self-Similarity (SS) Output Analysis:** Examine SS metrics and relate them to apnea indices and CPAP success.

## Installation

Ensure Python 3 is installed, then install the required dependencies:
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn h5py
```

## Scripts

### 1. `figures_piechart_histograms_heatmaps.py`
This script focuses on:
- Generating a pie chart to illustrate the distribution of high loop gain breathing oscillations (self-similarity) across different sleep stages.
- Creating histograms to examine how CPAP failure rates vary across ranges of the apnea-hypopnea index (AHI), central apnea index (CAI), and self-similarity (SS), based on the 3% hypopnea rule.
- Producing heatmaps for correlation and stratification analysis.
- Filtering data to remove bad signal recordings.

### 2. `figures_roc_prc_calibration.py`
This script computes:
- Receiver Operating Characteristic (ROC) curves and Precision-Recall (PR) curves for model evaluation.
- Calibration curves to measure probability estimation reliability.
- Logistic regression models with cross-validation to predict CPAP success or failure.

### 3. `SS_output_analysis.py`
This script analyzes self-similarity (SS) output data and provides:
- SS score computation and interpretation.
- Event extraction and classification.
- CPAP success determination based on SS metrics and apnea indices.

## Figures

- **Pie Chart (High Loop Gain Distribution):** Displays the proportion of high loop gain breathing oscillations (tagged SS events) in each sleep stage, offering a quick view of where these patterns predominantly occur.

- **Histograms (CPAP Failure vs. AHI, CAI, and SS):** Show the percentage of patients who fail CPAP therapy across different levels of the Apnea-Hypopnea Index (AHI), the Central Apnea Index (CAI), and self-similarity (SS). These are derived using the 3% hypopnea rule and include:
  - Red/blue bars highlighting the sex proportions in each bin.
  - The total number of patients contributing to each bin, shown above the bars.

- **ROC, PR, and Calibration Curves:** Evaluate the predictive power of CAI, hypoxic burden (Burden), and SS — individually and in combination — for determining CPAP failure under the 3% hypopnea rule. AUC values with 95% confidence intervals are included in the legends.

- **Heatmaps (SS vs. Hypoxic Burden):** Illustrate how self-similarity correlates with the hypoxic burden within CPAP success and failure cohorts. By comparing the two heatmaps side by side, users can quickly discern patterns and differences in these patient groups.

## Usage

Each script can be executed independently to generate the required visualizations and analyses. Ensure that the required data files (such as HDF5 and CSV datasets) are available in the expected directories.

Example usage for generating the pie chart, histograms, and heatmaps:
```bash
python figures_piechart_histograms_heatmaps.py
```

Example usage for computing ROC, PR, and calibration curves:
```bash
python figures_roc_prc_calibration.py
```

Example usage for analyzing SS output:
```bash
python SS_output_analysis.py
```


