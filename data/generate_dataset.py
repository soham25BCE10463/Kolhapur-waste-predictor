"""
Dataset Generator — Kolhapur Solid Waste Generation
Generates realistic synthetic data based on:
- Kolhapur has 66 wards, total population ~650,000 (2024 est.)
- KMC collects ~165 MT/day total
- Ward-level data derived from ward population proportions
- Seasonal, festival, weekday patterns based on real urban waste studies
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
import random

np.random.seed(42)
random.seed(42)
WARDS = {
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
    (1, 14): "Makar Sankranti",
    (1, 26): "Republic Day",
    (3, 8):  "Holi",
    (3, 25): "Gudhi Padwa",
    (4, 14): "Ambedkar Jayanti",
    (8, 15): "Independence Day",
    (8, 26): "Ganesh Chaturthi",
    (9, 5):  "Ganesh Visarjan",
    (10, 2): "Dussehra",
    (10, 24): "Diwali",
    (10, 25): "Diwali",
    (10, 26): "Diwali",
    (11, 1): "Post-Diwali",
    (12, 25): "Christmas",
    (12, 31): "New Year Eve",
}

def is_festival(d):
    return (d.month, d.day) in FESTIVALS

def season_factor(d):
    """Waste generation varies with season."""
    m = d.month
    if m in [11, 12, 1, 2]:
        return 1.0
    elif m in [3, 4, 5]:
        return 1.08
    elif m in [6, 7, 8, 9]:
        return 0.93
    else:
        return 1.12

def zone_base_rate(zone):
    """Commercial zones generate more waste per capita."""
    return {"Commercial": 0.38, "Mixed": 0.32, "Residential": 0.28, "Industrial": 0.30}[zone]

def generate_ward_data(ward_name, info, start_date, end_date):
    rows = []
    pop = info["population"]
    zone = info["zone"]
    base_rate = zone_base_rate(zone)

    current = start_date
    while current <= end_date:
        dow = current.weekday()

     
        weekday_factor = [1.02, 1.00, 1.00, 1.01, 1.08, 1.15, 0.92][dow]

        
        festival_factor = 1.35 if is_festival(current) else 1.0
        festival_name = FESTIVALS.get((current.month, current.day), "None")

  
        sf = season_factor(current)

        base_waste = pop * base_rate * weekday_factor * festival_factor * sf

        noise = np.random.normal(1.0, 0.05)
        waste_kg = max(0, base_waste * noise)

        if current.month in [6, 7, 8, 9]:
            rainfall_mm = max(0, np.random.normal(12, 8))
        else:
            rainfall_mm = max(0, np.random.normal(1, 2))

        temp_base = {1: 22, 2: 24, 3: 28, 4: 32, 5: 34, 6: 29,
                     7: 27, 8: 27, 9: 28, 10: 28, 11: 25, 12: 22}
        temp = temp_base[current.month] + np.random.normal(0, 2)

        rows.append({
            "date": current,
            "ward": ward_name,
            "zone_type": zone,
            "population": pop,
            "day_of_week": dow,
            "day_name": current.strftime("%A"),
            "month": current.month,
            "is_weekend": 1 if dow >= 5 else 0,
            "is_festival": 1 if festival_factor > 1 else 0,
            "festival_name": festival_name,
            "season_factor": round(sf, 3),
            "rainfall_mm": round(rainfall_mm, 1),
            "temperature_c": round(temp, 1),
            "waste_collected_kg": round(waste_kg, 1),
        })
        current += timedelta(days=1)
    return rows

start = date(2023, 1, 1)
end   = date(2024, 12, 31)

all_rows = []
for ward_name, info in WARDS.items():
    all_rows.extend(generate_ward_data(ward_name, info, start, end))

df = pd.DataFrame(all_rows)
df = df.sort_values(["date", "ward"]).reset_index(drop=True)

#Save
df.to_csv("data/kolhapur_waste_data.csv", index=False)
print(f"Dataset generated: {len(df)} rows, {df['ward'].nunique()} wards")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Avg daily total waste: {df.groupby('date')['waste_collected_kg'].sum().mean():.0f} kg")
print(df.dtypes)
