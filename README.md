# Kolhapur Solid Waste Generation Predictor

This is a machine learning system created by me that predicts daily ward level solid waste generation across Kolhapur Municipal Corporation (KMC) wards that will enable quick truck dispatch, fuel savings, and smarter resource allocations.



## Problem

Kolhapur generates approximately **165 metric tons of solid waste per day** across 66 wards. The KMC currently deploys collection trucks reactively, leading to:
1. Overflowing dustbins on festival days.
2. Wasted fuel and manpower on low waste days for eg:-Sundays, monsoon months, etc.
3. Missed collections in high density living wards

My project builds an ML model to **predict daily waste generation at the ward level** which is based on factors like days of week, seasons, festivals , rainfall.


## Results

| Model | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|
| Linear Regression | 459.7 kg | 617.1 kg | 0.8865 | 7.79% |
| **Random Forest** | **272.1 kg** | **381.7 kg** | **0.9566** | **4.46%** |

The Random Forest model achieves **R² of 0.9566** — explaining 95.6% of variance in daily waste generation.


## Project Structure

```
kolhapur-waste-predictor/
├── data/
│   ├── generate_dataset.py        # Realistic synthetic data generator
│   └── kolhapur_waste_data.csv    # Generated dataset (14,620 rows, 20 wards, 2 years)
├── notebooks/
│   └── Kolhapur_Waste_Predictor.ipynb  # Full walkthrough notebook
├── models/
│   ├── random_forest_model.pkl    # Trained Random Forest (best model)
│   ├── linear_regression_model.pkl
│   ├── label_encoder_zone.pkl
│   ├── label_encoder_ward.pkl
│   └── feature_names.pkl
├── plots/                         # All generated visualisations (8 plots)
├── outputs/
│   └── model_summary.json         # Model metrics summary
├── train_model.py                 # Full training script
├── predict.py                     # Prediction demo CLI
├── requirements.txt
└── README.md
```


## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/soham25BCE10463/Kolhapur-waste-predictor.git
cd kolhapur-waste-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the dataset

```bash
python data/generate_dataset.py
```

### 4. Train the models

```bash
python train_model.py
```

This will help you to:
- Train both Linear Regression and Random Forest models
- Print evaluation metrics to the terminal
- Save 8 visualisation plots to `plots/`
- Save trained models to `models/`

### 5. Run predictions

```bash
python predict.py
```


## Jupyter Notebook

Open the full walk through notebook for step by step EDA, model training, and interpretation:

```bash
jupyter notebook notebooks/Kolhapur_Waste_Predictor.ipynb
```

The notebook covers:
1. Data loading and exploration
2. Exploratory Data Analysis (EDA)
3. Feature engineering
4. Model training (Linear Regression + Random Forest)
5. Evaluation and comparison
6. Making predictions on new data
7. Key findings and conclusions


## Dataset

The dataset is generated synthetically but is grounded in real Kolhapur data:
- **20 real KMC wards** with approximate population figures from the 2011 Census extrapolated to 2024
- **Waste rates** calibrated to match KMC's reported ~165 MT/day city total
- **Festival calendar** based on real Maharashtra holidays (Makar Sankranti, Gudhi Padwa, Ganesh Chaturthi, Diwali, etc.)
- **Seasonal patterns** based on published urban solid waste studies for Tier-2 Maharashtra cities
- **Rainfall and temperature** distributions calibrated to Kolhapur's actual climate data


## Key Features Used

| Feature | Description |
|---|---|
| `population` | Ward population |
| `zone_type` | Commercial / Residential / Mixed / Industrial |
| `day_of_week` | 0 = Monday, 6 = Sunday |
| `is_weekend` | Binary flag |
| `is_festival` | Binary — major Maharashtra festivals |
| `post_festival` | Day after a festival (lagged feature) |
| `month` | Calendar month (1–12) |
| `season_factor` | Custom seasonal multiplier |
| `rainfall_mm` | Daily rainfall in mm |
| `temperature_c` | Daily temperature in °C |


## Key Findings

1. **Zone type and population** are the strongest predictors of waste volume
2. **Festival days see ~35% more waste** — the model captures this with the `is_festival` feature
3. **Monsoon months (June–September)** consistently generate ~7% less waste
4. **Saturdays** are peak collection days; Sundays the lowest
5. Random Forest captures **non-linear interactions** between features that Linear Regression misses — hence the significantly better R²


## Real-World Impact

If deployed by KMC, this system could:
- **Pre position extra trucks** on festival days preventing overflow
- **Reduce fuel costs ~15–18%** by scaling down Sunday/monsoon routes
- **Enable ward-priority dispatch** — commercial zones need daily collection, residential zones every other day during monsoon


## Requirements:-

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
joblib>=1.3
```

## Course Context

**Course:** Fundamentals of AI and ML  
**Project Type:** Bring Your Own Project (BYOP)  
**Institution:** VIT Bhopal University
**Problem domain:** Smart Cities / Urban Infrastructure / Environmental Technology


## Acknowledgements

- Kolhapur Municipal Corporation public data reports
- India Open Government Data Platform (data.gov.in) — SWM datasets
- Maharashtra Pollution Control Board — AQI & waste statistics
- Central Pollution Control Board — Municipal Solid Waste guidelines
