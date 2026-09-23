# Non-Intrusive-Load-Monitoring (NILM)

A machine-learning based **Non-Intrusive Load Monitoring (NILM)** system using electrical measurements from a **Siemens SENTRON PAC4200** power meter to identify individual appliances from aggregate power data.

> **Project result:** Random Forest achieved **100% accuracy (29/29)** on the held-out real measurement samples used in the project evaluation.

## Overview

NILM attempts to answer a simple question:

> **Which electrical devices are operating when only the aggregate electrical signal is measured?**

Instead of installing a dedicated energy meter on every appliance, this project analyzes aggregate electrical measurements and extracts electrical signatures that can be used to distinguish appliance types.

### System architecture

```mermaid
flowchart LR
    A[Siemens SENTRON PAC4200] --> B[Aggregate Electrical Measurements]
    B --> C[Preprocessing & Acquisition]
    C --> D[Feature Extraction]
    D --> E[12 Electrical Features]
    E --> F[Machine Learning]
    F --> G[Appliance Identification]
    G --> H[Online Tracker / Disaggregation]
    H --> I[Web Dashboard]
```

## Key results

The classification experiment was trained on synthetic feature data and evaluated on both synthetic test data and **held-out real PAC4200 recordings**.

| Model | Synthetic-test accuracy | Real accuracy | Correct real samples |
|---|---:|---:|---:|
| **Random Forest** | **100.0%** | **100.0%** | **29/29** |
| Logistic Regression | 100.0% | 75.9% | 22/29 |
| k-NN (k=5) | 99.7% | 58.6% | 17/29 |

The real evaluation contains **29 samples across 15 appliance classes**. The samples include repeated cycles from recordings, so 29/29 should not be interpreted as 29 statistically independent experiments.

### Most important features

The Random Forest feature-importance analysis identified the following features among the strongest predictors:

- `thd_i_mean`
- `q_mean`
- `p_mean`
- `s_mean`
- `qp_ratio`
- `p_std`
- `crest_factor`
- `thd_u_mean`

## Appliance classes

The classification pipeline covers the following measured appliance types:

- Coffee Machine
- Cooler Fan
- CPU
- Fluorescent Lamp
- Incandescent Light
- Laptop
- Monitor
- Table Fan
- Water Heater
- Hair Dryer (Foen)
- LED Lamp
- Fluorescent Tube
- Mixer
- Toaster
- USB Charger

PV is treated separately as a generating load and is excluded from the standard appliance-classification evaluation.

## Technical approach

### 1. Data acquisition

Electrical measurements are collected from a Siemens SENTRON PAC4200. The project works with quantities such as:

- Active power (`P`)
- Reactive power (`Q`)
- Apparent power (`S`)
- Power factor
- Voltage/current
- Voltage and current THD
- Frequency

### 2. Signal preprocessing

The acquisition pipeline converts the PAC4200 recordings into a consistent format and produces cleaned data for downstream analysis.

### 3. Feature engineering

The classifier uses electrical and statistical features derived from the measured signal, including power, reactive power, apparent power, THD, variability and waveform-related characteristics.

### 4. Synthetic-data generation

Because only a limited number of real recordings were available, the project generates synthetic training/test samples from measured appliance fingerprints.

Training and test randomness is controlled through NumPy random generators and recorded seeds.

### 5. Machine learning

Several models were compared:

- Random Forest
- Logistic Regression
- k-Nearest Neighbours

Random Forest was selected as the primary classifier because it performed best on the held-out real measurements and provides interpretable feature importance.

### 6. Online monitoring prototype

The repository also contains a prototype real-time layer:

```text
PAC4200 / recorded CSV
        ↓
Online step detector
        ↓
Running-set tracker
        ↓
Random Forest classifier
        ↓
Web dashboard
```

The web application supports recorded-data replay and a live PAC4200 Modbus TCP mode.

## Project structure

```text
non-intrusive-load-monitoring/
│
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── device_calibration.json
│
├── src/
│   └── nilm/
│       ├── acquisition.py
│       ├── aggregator.py
│       ├── calibration.py
│       ├── classification.py
│       ├── clustering.py
│       ├── disaggregator.py
│       ├── events.py
│       ├── features.py
│       ├── loadtype.py
│       ├── novelty.py
│       ├── onboarding.py
│       ├── paths.py
│       ├── registry.py
│       ├── steadystate.py
│       ├── streaming.py
│       ├── synthesizer.py
│       ├── tracker.py
│       ├── verification.py
│       └── visualization.py
│
├── notebooks/
│   ├── milestone1_notebook.ipynb
│   └── nilm_master.ipynb
│
├── outputs/
│   ├── appliance_fingerprints.json
│   ├── understanding_summary.csv
│   ├── figures/
│   └── reports/
│
├── data/
│   └── README.md
│
└── webapp/
    ├── app.py
    ├── index.html
    └── recordings/
        └── README.md
```

## Installation

Python **3.11** is recommended.

```bash
python -m venv .venv
```

### Windows

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux / macOS

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the analysis

From the project root:

### Windows PowerShell

```powershell
$env:PYTHONPATH="src"
python -m nilm.classification
```

Other pipeline modules include:

```powershell
python -m nilm.acquisition
python -m nilm.synthesizer
python -m nilm.verification
```

The notebooks provide the easiest way to explore the complete workflow and results.

## Running the web dashboard

```bash
cd webapp
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

For a real PAC4200 connection, set the meter address through an environment variable rather than hard-coding a machine-specific address:

```powershell
$env:PAC4200_HOST="192.168.168.1"
python app.py
```

The exact IP address depends on the local laboratory/network configuration.

## Data policy

The public repository intentionally does **not** include the full raw PAC4200 recordings, cleaned datasets, synthetic training/test datasets, or live recordings. These can be large and may be subject to university/project distribution restrictions.

The source code, notebooks, derived reports, figures and project documentation are included so that the methodology and results remain inspectable.

See [`data/README.md`](data/README.md).

## Limitations

- The real evaluation set is relatively small.
- Several real samples are repeated cycles from the same recordings.
- Synthetic data is generated from a limited number of measured appliance fingerprints.
- Synthetic-to-real transfer is therefore an important consideration.
- Live PAC4200 Modbus operation requires compatible hardware and the correct local register/network configuration.

## Project status

| Component | Status |
|---|---|
| PAC4200 data acquisition | Completed |
| Data preprocessing | Completed |
| Synthetic data generation | Completed |
| Feature engineering | Completed |
| Clustering analysis | Completed |
| Appliance classification | Completed |
| Real-data evaluation | Completed |
| Online step detection | Prototype |
| Running-set tracking | Prototype |
| Web dashboard | Prototype |

## Technologies

**Python · NumPy · Pandas · SciPy · Scikit-learn · Matplotlib · Jupyter · Flask · Modbus TCP · Siemens SENTRON PAC4200 · Signal Processing · Machine Learning**

## Author

**Mayank Dinesh Mehta**

M.Eng. Automation and IT  
Technische Hochschule Köln

GitHub: [@Mayank07-Git](https://github.com/Mayank07-Git)

## Project context

University project focused on electrical measurement analysis, signal processing, machine learning and non-intrusive appliance identification.
