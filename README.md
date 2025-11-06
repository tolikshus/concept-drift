# Concept Drift Experiments

**Authors**: Roie David, Anatoly Shusterman, Yossi Oren  
**Affiliation**: Department of Software and Information Systems Engineering (SISE), Ben-Gurion University of the Negev, Israel

This repository contains implementations and experiments based on the paper "Understanding and Addressing Concept Drift in Website Fingerprinting" by Roie David, Anatoly Shusterman, and Yossi Oren. The experiments are designed to reproduce and extend the results presented in the paper, focusing on various scenarios of concept drift in website fingerprinting attacks.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
- [Experiment Details](#experiment-details)
- [Results](#results)
- [References](#references)

## Project Overview
This project implements a series of experiments to study concept drift in website fingerprinting attacks, following the methodology and setups described in "Understanding and Addressing Concept Drift in Website Fingerprinting". The experiments cover baseline performance degradation, incremental learning, transfer learning, and various noise defense scenarios to analyze model performance under changing data distributions over time.

## Installation
1. Clone the repository:
   ```powershell
   git clone <repo-url>
   ```
2. Install required Python packages:
   ```powershell
   pip install -r requirements.txt
   ```

## Directory Structure
- `notebooks/`
  - `concept-drift-data-load.ipynb`: Data loading and preprocessing from Mendeley dataset repository.
  - `concept-drift-training.ipynb`: Initial model training (Model 0) using CNN-LSTM architecture.
  - `concept-drift-baseline.ipynb`: Baseline evaluation without adaptation (closed-world and open-world settings).
  - `concept-drift-incremental.ipynb`: Incremental learning with periodic model updates using new data.
  - `concept-drift-transfer.ipynb`: Transfer learning approach retraining model from scratch at each time period.
  - `concept-drift-baseline-noise.ipynb`: Baseline evaluation with parametric noise countermeasures.
  - `concept-drift-baseline-cache-noise.ipynb`: Baseline with cache-specific noise defense mechanisms.
  - `concept-drift-baseline-packet-noise.ipynb`: Baseline with packet-level noise defense mechanisms.
  - `nn_utils.py`: Neural network utility functions.
  - `utils.py`: General utility functions for data processing and model evaluation.
- `requirements.txt`: Python dependencies.

## Usage
1. **Data Loading**: Start by running `concept-drift-data-load.ipynb` to download and prepare the dataset from the Mendeley Data repository.
2. **Model Training**: Execute `concept-drift-training.ipynb` to train the initial CNN-LSTM model (Model 0) on the first time period.
3. **Run Experiments**: Choose one or more experiment notebooks based on your research goals:
   - For baseline performance: `concept-drift-baseline.ipynb`
   - For adaptation strategies: `concept-drift-incremental.ipynb` or `concept-drift-transfer.ipynb`
   - For noise countermeasures: `concept-drift-baseline-noise.ipynb`, `concept-drift-baseline-cache-noise.ipynb`, or `concept-drift-baseline-packet-noise.ipynb`
4. Modify hyperparameters and configuration settings as needed within each notebook.

## Experiment Details

### Data Loading (`concept-drift-data-load.ipynb`)
This notebook handles data acquisition from the Mendeley Data repository using API authentication. It downloads the website fingerprinting dataset containing traffic traces collected over multiple time periods, which forms the foundation for studying temporal concept drift in website fingerprinting attacks.

#### Dataset Description
The dataset contains two types of side-channel measurements for website fingerprinting:

**1. Cache-Contention Dataset**
- Measurements of cache contention during webpage rendering
- **Resolution**: 15,000 samples of cache contention recorded over 30 seconds
- Used for cache-based side-channel attacks

**2. Network-Packets Dataset**
- Measurements of packet sizes and direction for each packet during webpage rendering
- **Resolution**: Truncated network packet traces with maximum 3,000 values recorded over 30 seconds
- Used for network-based traffic analysis attacks

**Data Collection Modes:**

The measurements were created under three experimental settings to study different aspects of concept drift:

1. **Dynamic Browser Version** (26 weeks)
   - Changing webpage content over time
   - Dynamic browser versions (updates applied)
   - Studies drift from both webpage evolution and browser updates

2. **Static Browser Version** (26 weeks)
   - Changing webpage content over time
   - Fixed browser version (no updates)
   - Isolates drift from webpage content changes only

3. **Static Webpages + Dynamic Browser** (2 weeks)
   - Fixed webpage content
   - Changing browser versions
   - Isolates drift from browser version changes only

### Model Training (`concept-drift-training.ipynb`)
Trains the initial CNN-LSTM neural network (Model 0) on the first time period. 
The model uses:
- **Cache attack**: 15,000 features, 256 features per layer, pool size 4, 32 LSTM units
- **Network attack**: 3,000 features, 256 features per layer, pool size 3, 128 LSTM units
- **Classifier**: 100 website classes (closed-world) or 70 classes + open-world category
- **Training**: Batch size 256, early stopping with patience 5, Adam optimizer

### Baseline Experiments (`concept-drift-baseline.ipynb`)
Evaluates Model 0's performance across all subsequent time periods without any adaptation, establishing the baseline degradation due to temporal concept drift. Tests both:
- **Closed-world setting**: 100 known websites
- **Open-world setting**: 70 monitored + 30 unmonitored websites (threshold-based classification) 

This demonstrates how model accuracy deteriorates over time as data distributions shift.

### Incremental Learning (`concept-drift-incremental.ipynb`)
Implements online adaptation strategy where the model is continuously fine-tuned with small amounts of new data from each time period:
- Starts from the base Model 0 trained in the initial training stage
- Collects n samples per website (default: 10-20) from the current time period
- Fine-tunes the existing model with learning rate 1e-5 for up to 20 epochs
- Updates model weights progressively, carrying forward learned knowledge from previous time periods
- Evaluates if continuous adaptation can mitigate concept drift

### Transfer Learning (`concept-drift-transfer.ipynb`)
Implements periodic retraining strategy where the model is retrained from the base model at each time period:
- Starts from the base Model 0 trained in the initial training stage
- Takes n samples per website (default: 10) from the current time period
- Clones Model 0 architecture and resets to the original base model weights (not random initialization)
- Trains for up to 30 epochs with learning rate 1e-5
- Tests whether periodic retraining from the stable base model is more effective than incremental updates

### Noise Defense Experiments

#### General Noise (`concept-drift-baseline-noise.ipynb`)
Evaluates baseline model robustness against generic countermeasures:
- **Cache noise**: Gaussian noise with varying standard deviations.
- **Network noise**: Packet insertion with varying noise std and insertion probabilities.
- Tests model performance degradation under different defense intensity levels

#### Cache-Specific Noise (`concept-drift-baseline-cache-noise.ipynb`)
Focused evaluation of cache-based countermeasures using absolute value Gaussian noise applied to cache access patterns, testing multiple noise standard deviations to assess defense-accuracy tradeoffs.

#### Packet-Specific Noise (`concept-drift-baseline-packet-noise.ipynb`)
Focused evaluation of network-based countermeasures using packet insertion and timing perturbations. Tests various combinations of noise intensity and insertion probability to understand the impact on classifier accuracy.

## Results
Results and outputs are generated within each notebook as the experiments run:
- **Accuracy metrics**: Per-date accuracy scores tracking model performance over time
- **Visualization plots**: Time-series plots showing accuracy degradation or adaptation patterns
- **Comparative analysis**: Performance comparisons across different strategies (baseline vs. incremental vs. transfer)
- **Defense effectiveness**: Accuracy under various noise countermeasure intensities

All metrics are computed using test data from each time period and stored in results dictionaries within the notebooks.


## References
- **Roie David, Anatoly Shusterman, Yossi Oren**. "Understanding and Addressing Concept Drift in Website Fingerprinting". *Elsevier Computer Networks Journal*, to appear.
- **Anatoly Shusterman, Roie David, Yossi Oren** (2025). "Concept Drift in Website Fingerprinting". *Mendeley Data*, V2. doi: [10.17632/fd6ggttgj4.2](https://doi.org/10.17632/fd6ggttgj4.2)

