# AthleteIQ — Sports Performance Telemetry Analytics

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Tableau](https://img.shields.io/badge/Tableau-Public-orange) ![ML](https://img.shields.io/badge/ML-RandomForest-green) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## Overview
End-to-end data analytics project simulating a real-world sports telemetry pipeline for a 10-player football squad across 15 matches. Covers data engineering, exploratory analysis, KPI development, machine learning, and an interactive Tableau dashboard.

## Live Dashboard
[View on Tableau Public](https://public.tableau.com/app/profile/shivansh.pandey1813/viz/AthleteIQSportsPerformanceTelemetryDashboard/AthleteIQSportsPerformanceTelemetryDashboard?publish=yes)

## Project Structure
AthleteIQ/
├── data/
│   ├── telemetry_raw.csv              # Simulated GPS + biometric data (13,500 rows)
│   ├── telemetry_processed.csv        # Cleaned data post-EDA
│   ├── telemetry_featured.csv         # With engineered features
│   ├── match_kpi_summary.csv          # Aggregated KPIs per player per match
│   └── match_kpi_with_predictions.csv # With injury risk predictions
├── scripts/
│   ├── generate_data.py               # Telemetry data simulation
│   ├── feature_engineering.py         # KPI calculation
│   └── model_injury_risk.py           # Random Forest + SHAP
├── reports/
│   ├── plot_01_speed_distribution.png
│   ├── plot_02_fatigue_curve.png
│   ├── plot_03_hr_vs_speed.png
│   ├── plot_04_speed_zones.png
│   ├── plot_05_correlation.png
│   ├── plot_06_injury_risk.png
│   ├── plot_07_confusion_matrix.png
│   ├── plot_08_feature_importance.png
│   └── plot_09_shap_summary.png
└── notebooks/
└── AthleteIQ_EDA.ipynb            # Full EDA notebook

## Key Features
- **13,500 rows** of simulated telemetry: GPS speed, heart rate, acceleration, sprint flags
- **6 engineered KPIs**: Fatigue Index, High Intensity %, Speed Drop-off, Total Distance, Sprint Count, Injury Risk Score
- **Random Forest model** predicting injury risk with ROC-AUC of 0.963 (5-fold CV)
- **SHAP explainability** — identifies top drivers of injury risk per player
- **Tableau dashboard** with 5 interactive charts

## Tech Stack
| Area | Tools |
|------|-------|
| Data wrangling | pandas, numpy |
| Visualization | matplotlib, seaborn, Tableau |
| Machine learning | scikit-learn (Random Forest) |
| Explainability | SHAP |
| Notebook | Google Colab |
| Version control | Git, GitHub |

## Key Insights
- **Rahul Das** (age 30, Midfielder) consistently flags highest injury risk across all 15 matches
- **Forwards** average 30.87 sprints per match vs 8.84 for Defenders
- **Nikhil Rao** shows the steepest speed drop-off (30.4%) — strongest fatigue signature in the squad
- **Karan Singh** leads high intensity zone time at 55.6% of match duration

## Setup
```bash
git clone https://github.com/ShivanshPandey07/AthleteIQ.git
cd AthleteIQ
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy matplotlib seaborn scikit-learn shap
python scripts/generate_data.py
python scripts/feature_engineering.py
python scripts/model_injury_risk.py
```

## Author
**Shivansh Pandey** 
