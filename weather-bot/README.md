# Weather Forecast Pipeline

A production-style machine learning pipeline that generates daily maximum temperature forecasts for 45 cities worldwide. The system ingests real-time METAR observations and NWP ensemble model data, trains a gradient boosting model, applies a real-time bias corrector, and outputs probabilistic forecasts with confidence intervals and bet recommendations for weather derivative markets.

---

## What It Does

Given a city name, the pipeline:

1. Loads and cleans historical METAR observations and NWP model forecasts
2. Engineers a feature matrix from ensemble model outputs, observed lags, moisture indices, solar radiation, and wind components
3. Trains a Gradient Boosting or Ridge regression model depending on data availability
4. Seeds a real-time exponential bias corrector from out-of-sample walk-forward errors
5. Builds a synthetic forecast row for tomorrow using tomorrow's NWP data and today's observed lags
6. Outputs a final temperature forecast with 80% and 95% confidence intervals
7. Computes bucket probabilities and Kelly criterion bet recommendations against market prices

---

## Project Structure

```
weather-bot/
├── Data/
│   ├── metar_data/          # METAR observation CSVs per station
│   ├── forcast_data/        # NWP model forecast CSVs per city
│   └── corrector-folder-v4/ # Persisted bias corrector state
│
└── mos_pipeline/
    ├── main.py              # Entry point — city selection, updater, pipeline
    ├── pipeline.py          # Orchestrates all stages end to end
    ├── config.py            # Cities, model list, constants
    ├── data.py              # Data loading, cleaning, alignment
    ├── features.py          # Feature engineering
    ├── model.py             # Model training, walk-forward validation, cross-validation
    ├── forecast.py          # Forecast generation, bucket probabilities, bet recommendations
    ├── corrector.py         # Real-time bias corrector with exponential decay
    ├── update.py            # Data updater — fetches latest METAR and NWP data
    ├── download_data.py     # Downloads historical data from S3
    └── requirements.txt
```

---

## Pipeline Stages

```
METAR CSVs ──┐
             ├──► Clean & Align ──► Feature Engineering ──► Train Model
NWP CSVs ────┘                                                    │
                                                                  ▼
                                                     Seed Bias Corrector
                                                                  │
                                                                  ▼
                                                    Build Tomorrow's Row
                                                                  │
                                                                  ▼
                                                     Final Forecast + CI
                                                                  │
                                                                  ▼
                                                    Bucket Probs + Bets
```

---

## Feature Groups

The model uses 13 feature groups covering:

- **NWP ensemble temperatures** — 6 model outputs (ECMWF IFS, ECMWF IFS025, GFS, GEM, ICON, UKMO)
- **Ensemble statistics** — spread, mean, standard deviation, max spread
- **Observed lags** — yesterday, 2 days ago, 3/5/7-day rolling means and anomalies
- **Per-model rolling bias** — 3-day (fast regime) and 14-day (seasonal drift) error trackers
- **Moisture** — dewpoint depression, observed dewpoint, humidity, 3-day moisture trends
- **Solar/cloud** — GHI, effective solar, cloud-adjusted solar efficiency, clear-calm index
- **Wind** — speed, U/V directional components (city-specific sea-breeze and Föhn effects)
- **Apparent temperature** — Australian BOM formula applied to METAR observations
- **Calendar** — day of year, month

---

## Real-Time Bias Corrector

The corrector sits on top of the ML model and captures regime-dependent errors the training data cannot:

- Exponentially weighted mean of recent out-of-sample errors (default window: 5 days, decay: 0.7)
- Jump dampening using `exp(-jump / scale)` — prevents over-correction during heat spikes or cold outbreaks
- Seeded from genuine out-of-sample walk-forward predictions, not in-sample fits
- State persisted to disk between runs so operational errors accumulate over time

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/faizssmohammad/weather-bot
cd weather-bot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r mos_pipeline/requirements.txt
```

### 4. Download historical data from S3

```bash
python mos_pipeline/download_data.py
```

This downloads all METAR and NWP forecast CSVs into the `Data/` folder (~XX MB). No AWS account or credentials required — the bucket is public.

### 5. Run the pipeline

```bash
python mos_pipeline/main.py london
```

---

## Updating Data

To fetch the latest observations and model runs before running the pipeline:

```bash
python mos_pipeline/main.py london
```

The updater runs automatically before the pipeline — it checks the last date in the forecast CSV and fetches everything from that date to today.

---

## Supported Cities

45 cities across 6 continents including London, New York, Tokyo, Beijing, Mumbai, Sydney, São Paulo, Dubai, and more. Full list in `mos_pipeline/config.py`.

---

## Output Example

```
=================================================================
  MOS  |  EGLC  |  london  |  Europe/London
=================================================================
[6] FORECAST FOR 2026-06-08
  Target date          : 2026-06-08
  ML model forecast    : 19.4C
  Last observed tmax   : 18.1C
  Forecast jump        : +1.3C
  ──────────────────────────────────────────────
  Real-time correction : +0.42C
  ──────────────────────────────────────────────
  Final forecast       : 19.8C
  80% CI               : 17.9C – 21.7C
  95% CI               : 16.9C – 22.7C

[8] BET RECOMMENDATION
  Bucket   ML%     Market%    Edge    Action
  19C      18.4%   13.1%     +5.3%   BET
  20C      21.2%   27.8%     -6.6%   BET NO
```

---

## Tech Stack

- **Python 3.11**
- **pandas / numpy** — data processing
- **scikit-learn** — GradientBoostingRegressor, Ridge, StandardScaler
- **scipy** — normal distribution for probabilistic bucket forecasts
- **pytz** — timezone handling
- **boto3** — S3 data download
- **requests / requests-cache** — METAR and NWP API calls

---

## Data Sources

- **METAR observations** — Iowa Environmental Mesonet (IEM) ASOS network
- **NWP model forecasts** — Open-Meteo API (ECMWF IFS, ECMWF IFS025, GFS Seamless, GEM Seamless, ICON Seamless, UKMO Seamless)