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

## Publication
> Nassi TE, Oppersma E, Labarca G, Donker DW, Westover MB, Thomas RJ. 
> Morphological Prediction of CPAP Associated Acute Respiratory Instability. 
> Ann Am Thorac Soc. 2024 Sep 17;22(1):138–49. 
> doi: 10.1513/AnnalsATS.202311-979OC. 
> Epub ahead of print. PMID: 39288402; PMCID: PMC11708763.