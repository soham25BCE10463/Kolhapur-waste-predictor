"""
Kolhapur Solid Waste Generation Predictor
ML Training Script — Linear Regression + Random Forest Comparison
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

#Style
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})
COLORS = ['#4A90D9', '#E87040', '#5BAD6F', '#9B59B6', '#E74C3C', '#F39C12']
print("="*60)
print("KOLHAPUR SOLID WASTE PREDICTOR — MODEL TRAINING")
print("="*60)

df = pd.read_csv("data/kolhapur_waste_data.csv", parse_dates=["date"])
print(f"\n✓ Loaded {len(df):,} records | {df['ward'].nunique()} wards | "
      f"{df['date'].dt.year.nunique()} years\n")

le_zone = LabelEncoder()
le_ward = LabelEncoder()
df['zone_encoded'] = le_zone.fit_transform(df['zone_type'])
df['ward_encoded'] = le_ward.fit_transform(df['ward'])
df['quarter'] = df['date'].dt.quarter
df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
df['is_monday'] = (df['day_of_week'] == 0).astype(int)
df['post_festival'] = df['is_festival'].shift(1).fillna(0).astype(int)
df['pop_density_proxy'] = df['population'] / 1000  # normalized

FEATURES = [
    'population', 'zone_encoded', 'ward_encoded',
    'day_of_week', 'is_weekend', 'is_festival', 'post_festival',
    'month', 'quarter', 'week_of_year', 'is_monday',
    'season_factor', 'rainfall_mm', 'temperature_c',
]
TARGET = 'waste_collected_kg'

X = df[FEATURES]
y = df[TARGET]

#Train
split_date = df['date'].max() - pd.Timedelta(days=90)
train_mask = df['date'] <= split_date
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]
df_test = df[~train_mask].copy()

print(f"Training samples : {len(X_train):,}")
print(f"Test samples     : {len(X_test):,}  (last 90 days)")
print("\n--- Training Linear Regression ---")
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("--- Training Random Forest ---")
rf = RandomForestRegressor(n_estimators=200, max_depth=12,
                           min_samples_leaf=5, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)


def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"\n{name}")
    print(f"  MAE  : {mae:.1f} kg  (avg error per ward-day)")
    print(f"  RMSE : {rmse:.1f} kg")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAPE : {mape:.2f}%")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

print("\n" + "="*40)
print("MODEL EVALUATION RESULTS")
print("="*40)
lr_metrics = evaluate("Linear Regression", y_test, lr_pred)
rf_metrics = evaluate("Random Forest    ", y_test, rf_pred)

joblib.dump(rf, "models/random_forest_model.pkl")
joblib.dump(lr, "models/linear_regression_model.pkl")
joblib.dump(le_zone, "models/label_encoder_zone.pkl")
joblib.dump(le_ward, "models/label_encoder_ward.pkl")
joblib.dump(FEATURES, "models/feature_names.pkl")
print("\n✓ Models saved to models/")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Kolhapur Waste Predictor — Actual vs Predicted", fontsize=14, y=1.02)

for ax, pred, name, color in zip(
    axes, [lr_pred, rf_pred],
    ["Linear Regression", "Random Forest"],
    [COLORS[0], COLORS[1]]
):
    ax.scatter(y_test, pred, alpha=0.3, s=8, color=color)
    mn, mx = y_test.min(), y_test.max()
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1, label='Perfect fit')
    ax.set_xlabel("Actual Waste (kg)")
    ax.set_ylabel("Predicted Waste (kg)")
    ax.set_title(f"{name}\nR² = {r2_score(y_test, pred):.4f} | MAPE = {np.mean(np.abs((y_test-pred)/y_test))*100:.1f}%")
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("plots/01_actual_vs_predicted.png", bbox_inches='tight')
plt.close()
print("✓ Plot 1 saved")

importances = rf.feature_importances_
fi_df = pd.DataFrame({'feature': FEATURES, 'importance': importances})
fi_df = fi_df.sort_values('importance', ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(fi_df['feature'], fi_df['importance'],
               color=[COLORS[2] if i > len(FEATURES)-4 else '#B0C4DE'
                      for i in range(len(fi_df))], height=0.7)
ax.set_xlabel("Feature Importance (Gini)")
ax.set_title("What Drives Waste Generation?\nRandom Forest Feature Importance")
ax.axvline(fi_df['importance'].mean(), color='gray', ls='--', lw=1,
           label=f'Mean ({fi_df["importance"].mean():.3f})')
ax.legend()
plt.tight_layout()
plt.savefig("plots/02_feature_importance.png", bbox_inches='tight')
plt.close()
print("✓ Plot 2 saved")

df_test = df_test.copy()
df_test['rf_pred'] = rf_pred

daily_actual = df_test.groupby('date')['waste_collected_kg'].sum() / 1000  # MT
daily_pred   = df_test.groupby('date')['rf_pred'].sum() / 1000

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(daily_actual.index, daily_actual.values, label='Actual', color=COLORS[0], lw=1.5)
ax.plot(daily_pred.index,   daily_pred.values,   label='Predicted (RF)',
        color=COLORS[1], lw=1.5, ls='--')
ax.fill_between(daily_actual.index,
                daily_actual.values, daily_pred.values,
                alpha=0.15, color=COLORS[1])
ax.set_xlabel("Date")
ax.set_ylabel("Total Waste (Metric Tons / day)")
ax.set_title("City-Wide Daily Waste: Actual vs Predicted\n(Test period — last 90 days)")
ax.legend()
plt.tight_layout()
plt.savefig("plots/03_citywide_timeseries.png", bbox_inches='tight')
plt.close()
print("✓ Plot 3 saved")

dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
dow_avg = df.groupby('day_of_week')['waste_collected_kg'].mean()

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(dow_names, dow_avg.values,
              color=[COLORS[1] if i >= 5 else COLORS[0] for i in range(7)], width=0.6)
ax.set_ylabel("Avg Waste per Ward (kg)")
ax.set_title("Waste Generation by Day of Week\nSaturdays peak; Sundays dip")
weekend_patch = mpatches.Patch(color=COLORS[1], label='Weekend')
weekday_patch = mpatches.Patch(color=COLORS[0], label='Weekday')
ax.legend(handles=[weekday_patch, weekend_patch])
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig("plots/04_waste_by_weekday.png", bbox_inches='tight')
plt.close()
print("✓ Plot 4 saved")

pivot = df.pivot_table(values='waste_collected_kg',
                       index='ward', columns='month', aggfunc='mean')
month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec']
pivot.columns = month_labels

fig, ax = plt.subplots(figsize=(14, 7))
sns.heatmap(pivot, ax=ax, cmap='YlOrRd', linewidths=0.3,
            cbar_kws={'label': 'Avg Waste (kg/day)'}, annot=False)
ax.set_title("Ward × Month Waste Heatmap\nKolhapur Municipal Corporation", fontsize=13)
ax.set_xlabel("Month")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig("plots/05_ward_month_heatmap.png", bbox_inches='tight')
plt.close()
print("✓ Plot 5 saved")

zone_df = df.groupby(['zone_type', 'month'])['waste_collected_kg'].mean().reset_index()
fig, ax = plt.subplots(figsize=(10, 4))
for i, zone in enumerate(zone_df['zone_type'].unique()):
    zd = zone_df[zone_df['zone_type'] == zone]
    ax.plot(zd['month'], zd['waste_collected_kg'], marker='o',
            label=zone, color=COLORS[i], lw=2)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_labels)
ax.set_ylabel("Avg Waste per Ward (kg/day)")
ax.set_title("Seasonal Waste Trends by Zone Type")
ax.legend()
plt.tight_layout()
plt.savefig("plots/06_zone_seasonal_trends.png", bbox_inches='tight')
plt.close()
print("✓ Plot 6 saved")

metrics_df = pd.DataFrame({
    'Metric': ['MAE (kg)', 'RMSE (kg)', 'R² (×1000)', 'MAPE (%)'],
    'Linear Regression': [lr_metrics['MAE'], lr_metrics['RMSE'],
                          lr_metrics['R2']*1000, lr_metrics['MAPE']],
    'Random Forest':     [rf_metrics['MAE'], rf_metrics['RMSE'],
                          rf_metrics['R2']*1000, rf_metrics['MAPE']],
})

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
fig.suptitle("Model Comparison — Linear Regression vs Random Forest", fontsize=13)
for ax, (_, row) in zip(axes, metrics_df.iterrows()):
    vals = [row['Linear Regression'], row['Random Forest']]
    bars = ax.bar(['LR', 'RF'], vals,
                  color=[COLORS[0], COLORS[1]], width=0.5)
    ax.set_title(row['Metric'])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                f'{v:.2f}', ha='center', fontsize=10)
    ax.set_ylim(0, max(vals) * 1.25)

plt.tight_layout()
plt.savefig("plots/07_model_comparison.png", bbox_inches='tight')
plt.close()
print("✓ Plot 7 saved")

fest_avg   = df[df['is_festival']==1]['waste_collected_kg'].mean()
normal_avg = df[df['is_festival']==0]['waste_collected_kg'].mean()

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(['Normal Days', 'Festival Days'], [normal_avg, fest_avg],
       color=[COLORS[0], COLORS[3]], width=0.5)
ax.set_ylabel("Avg Waste per Ward (kg/day)")
ax.set_title(f"Festival Effect on Waste\n+{((fest_avg/normal_avg)-1)*100:.1f}% more waste on festival days")
for x, v in enumerate([normal_avg, fest_avg]):
    ax.text(x, v + 20, f'{v:.0f} kg', ha='center', fontsize=11)
plt.tight_layout()
plt.savefig("plots/08_festival_impact.png", bbox_inches='tight')
plt.close()
print("✓ Plot 8 saved")

print("\n" + "="*60)
print("ALL DONE — model training + 8 plots complete")
print(f"Best model: Random Forest (R² = {rf_metrics['R2']:.4f}, MAPE = {rf_metrics['MAPE']:.2f}%)")
print("="*60)
