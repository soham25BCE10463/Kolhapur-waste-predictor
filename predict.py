"""
predict.py — Kolhapur Solid Waste Predictor
Usage: python predict.py

Interactive CLI to predict waste generation for any ward on any date.
"""

import joblib
import numpy as np
import pandas as pd
from datetime import date, datetime

# Load saved artefacts
rf        = joblib.load("models/random_forest_model.pkl")
le_zone   = joblib.load("models/label_encoder_zone.pkl")
le_ward   = joblib.load("models/label_encoder_ward.pkl")
FEATURES  = joblib.load("models/feature_names.pkl")

WARD_INFO = {
    "Rajaram Road":        {"population": 18200, "zone": "Commercial"},
    "Mahadwar Road":       {"population": 15400, "zone": "Commercial"},
    "Shivaji Peth":        {"population": 21000, "zone": "Mixed"},
    "Kasba Bawada":        {"population": 19800, "zone": "Residential"},
    "Tarabai Park":        {"population": 14200, "zone": "Residential"},
    "Shahupuri":           {"population": 22500, "zone": "Commercial"},
    "Laxmipuri":           {"population": 17800, "zone": "Mixed"},
    "Mangalwar Peth":      {"population": 20100, "zone": "Commercial"},
    "Rajarampuri":         {"population": 23400, "zone": "Residential"},
    "Udyam Nagar":         {"population": 16500, "zone": "Industrial"},
    "Phulewadi":           {"population": 18700, "zone": "Residential"},
    "Kothali":             {"population": 12300, "zone": "Residential"},
    "Nagala Park":         {"population": 19200, "zone": "Mixed"},
    "Gandhinagar":         {"population": 17600, "zone": "Residential"},
    "Sambhajinagar":       {"population": 21800, "zone": "Mixed"},
    "Line Bazar":          {"population": 13900, "zone": "Commercial"},
    "Rankala":             {"population": 16800, "zone": "Residential"},
    "Karveer":             {"population": 14500, "zone": "Residential"},
    "Balinge":             {"population": 11200, "zone": "Residential"},
    "Mhasve":              {"population": 10800, "zone": "Residential"},
}

FESTIVALS = {
    (1, 14), (1, 26), (3, 8), (3, 25), (4, 14),
    (8, 15), (8, 26), (9, 5), (10, 2), (10, 24),
    (10, 25), (10, 26), (11, 1), (12, 25), (12, 31),
}

def season_factor(d):
    m = d.month
    if m in [11, 12, 1, 2]: return 1.0
    elif m in [3, 4, 5]:    return 1.08
    elif m in [6, 7, 8, 9]: return 0.93
    else:                   return 1.12

def build_input(ward_name, pred_date, rainfall_mm=0.0, temperature_c=None):
    info = WARD_INFO[ward_name]
    pop  = info["population"]
    zone = info["zone"]

    #Default temperatur based on month
    temp_base = {1:22,2:24,3:28,4:32,5:34,6:29,7:27,8:27,9:28,10:28,11:25,12:22}
    if temperature_c is None:
        temperature_c = temp_base[pred_date.month]

    zone_enc = le_zone.transform([zone])[0]
    ward_enc = le_ward.transform([ward_name])[0]
    dow      = pred_date.weekday()
    month    = pred_date.month
    is_fest  = 1 if (pred_date.month, pred_date.day) in FESTIVALS else 0
    sf       = season_factor(pred_date)

    row = {
        'population':      pop,
        'zone_encoded':    zone_enc,
        'ward_encoded':    ward_enc,
        'day_of_week':     dow,
        'is_weekend':      1 if dow >= 5 else 0,
        'is_festival':     is_fest,
        'post_festival':   0,
        'month':           month,
        'quarter':         (month - 1) // 3 + 1,
        'week_of_year':    pred_date.isocalendar()[1],
        'is_monday':       1 if dow == 0 else 0,
        'season_factor':   sf,
        'rainfall_mm':     rainfall_mm,
        'temperature_c':   temperature_c,
    }
    return pd.DataFrame([row])[FEATURES]

def predict_waste(ward_name, pred_date, rainfall_mm=0.0, temperature_c=None):
    X = build_input(ward_name, pred_date, rainfall_mm, temperature_c)
    pred_kg = rf.predict(X)[0]
    return pred_kg

def predict_city_total(pred_date, rainfall_mm=0.0, temperature_c=None):
    total = 0.0
    results = {}
    for ward in WARD_INFO:
        kg = predict_waste(ward, pred_date, rainfall_mm, temperature_c)
        results[ward] = kg
        total += kg
    return total, results

#demo
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Kolhapur Solid Waste Predictor — Prediction Demo")
    print("="*55)

    demo_dates = [
        date(2025, 1, 14),   #Makar Sankranti
        date(2025, 3, 17),   #Normal weekday
        date(2025, 8, 15),   #Independence Day
        date(2025, 10, 24),  #Diwali
        date(2025, 6, 25),   #Monsoon weekday
    ]

    print("\n--- Single Ward Predictions (Shahupuri) ---")
    for d in demo_dates:
        kg = predict_waste("Shahupuri", d)
        print(f"  {d}  ({d.strftime('%A'):9})  →  {kg:,.0f} kg")

    print("\n--- City-Wide Total Predictions ---")
    for d in demo_dates[:3]:
        total, _ = predict_city_total(d)
        is_fest = "🎉 Festival" if (d.month, d.day) in FESTIVALS else ""
        print(f"  {d}  →  {total/1000:5.1f} MT  {is_fest}")

    print("\n--- Ward-by-Ward for Tomorrow ---")
    tomorrow = date(2025, 4, 1)
    total, ward_preds = predict_city_total(tomorrow)
    sorted_wards = sorted(ward_preds.items(), key=lambda x: x[1], reverse=True)
    print(f"  Date: {tomorrow}")
    print(f"  {'Ward':<22} {'Zone':<14} {'Predicted (kg)':>14}")
    print(f"  {'-'*52}")
    for ward, kg in sorted_wards:
        zone = WARD_INFO[ward]['zone']
        print(f"  {ward:<22} {zone:<14} {kg:>12,.0f}")
    print(f"  {'-'*52}")
    print(f"  {'CITY TOTAL':<36} {total:>12,.0f} kg  ({total/1000:.1f} MT)")
    print()
