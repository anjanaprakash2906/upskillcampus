# ============================================================
# Project 4: Prediction of Agriculture Crop Production in India
# ============================================================
# Dataset: Agriculture Production in India (2001-2014)
# Goal   : Predict (1) Crop Production Quantity
#                   (2) Cultivation Cost
# ============================================================

# ─────────────────────────────────────────────
# 0. IMPORTS & SETUP
# ─────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
try:
    import tkinter  # noqa: F401
    matplotlib.use('TkAgg')
except Exception:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.chdir(BASE_DIR)

# Plot style
sns.set_theme(style='whitegrid', palette='Set2')
plt.rcParams.update({'figure.dpi': 120, 'font.size': 11})
COLORS = sns.color_palette('Set2', 10)

print("=" * 60)
print("  Crop production prediction for India")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. LOAD DATASETS
# ─────────────────────────────────────────────
print("\n[1] Loading data files...")

# Primary dataset: Cost & Yield by Crop and State
BASE = BASE_DIR

cost_df = pd.read_csv(os.path.join(BASE, 'datafile (1).csv'))
cost_df.columns = ['Crop', 'State',
                   'Cost_Cultivation_A2FL', 'Cost_Cultivation_C2',
                   'Cost_Production_C2', 'Yield_Quintal_Hectare']

# Production / Area / Yield across years (2006-2011)
prod_area_df = pd.read_csv(os.path.join(BASE, 'datafile (2).csv'))
prod_area_df.columns = prod_area_df.columns.str.strip()
prod_area_df.rename(columns={'Crop             ': 'Crop'}, inplace=True)
prod_area_df.columns = prod_area_df.columns.str.strip()

# Variety & Season info
variety_df = pd.read_csv(os.path.join(BASE, 'datafile (3).csv'))
variety_df = variety_df[['Crop', 'Variety', 'Season/ duration in days', 'Recommended Zone']]
variety_df.columns = ['Crop', 'Variety', 'Season', 'Recommended_Zone']

# Production index (2004-2012)
index_df = pd.read_csv(os.path.join(BASE, 'datafile.csv'))

# Macro production stats (1993-2014)
macro_df = pd.read_csv(os.path.join(BASE, 'produce.csv'))

print(f"  Production cost dataset: {cost_df.shape}")
print(f"  Crop area dataset: {prod_area_df.shape}")
print(f"  Variety and season dataset: {variety_df.shape}")
print(f"  Index dataset: {index_df.shape}")
print(f"  Macro production dataset: {macro_df.shape}")

# ─────────────────────────────────────────────
# 2. DATA CLEANING & PREPROCESSING
# ─────────────────────────────────────────────
print("\n[2] Cleaning and preparing the data...")

# --- cost_df ---
print(f"  Missing values in the main crop dataset:\n{cost_df.isnull().sum()}")
cost_df.dropna(inplace=True)
cost_df['Crop']  = cost_df['Crop'].str.strip().str.upper()
cost_df['State'] = cost_df['State'].str.strip().str.title()

# --- prod_area_df ---
prod_area_df['Crop'] = prod_area_df['Crop'].str.strip()
prod_area_df.replace('NA', np.nan, inplace=True)
for c in prod_area_df.columns[1:]:
    prod_area_df[c] = pd.to_numeric(prod_area_df[c], errors='coerce')

# --- variety_df ---
variety_df.dropna(subset=['Crop'], inplace=True)
variety_df['Crop'] = variety_df['Crop'].str.strip().str.upper()
variety_df['Season'] = variety_df['Season'].str.strip()

# --- macro_df ---
year_cols = [c for c in macro_df.columns if c.startswith('3-')]
macro_df[year_cols] = macro_df[year_cols].replace('NA', np.nan)
for c in year_cols:
    macro_df[c] = pd.to_numeric(macro_df[c], errors='coerce')

print("  Data cleaning is done.")
print(f"  Cleaned crop dataset has: {cost_df.shape[0]} rows")

# ─────────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────
print("\n[3] Exploratory Data Analysis ...")

# ── 3.1 Cost & Yield stats ──────────────────
print("\n  -- Cost & Yield Summary --")
print(cost_df.describe().round(2))

# ── 3.2 Top crops by cultivation cost ───────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

top_crops_cost = (cost_df.groupby('Crop')['Cost_Cultivation_C2']
                  .mean().sort_values(ascending=False).head(12))
axes[0].barh(top_crops_cost.index[::-1], top_crops_cost.values[::-1],
             color=COLORS[0])
axes[0].set_title('Top 12 Crops by Avg Cultivation Cost (C2, ₹/Hectare)', fontweight='bold')
axes[0].set_xlabel('Cost (₹/Hectare)')
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))

top_crops_yield = (cost_df.groupby('Crop')['Yield_Quintal_Hectare']
                   .mean().sort_values(ascending=False).head(12))
axes[1].barh(top_crops_yield.index[::-1], top_crops_yield.values[::-1],
             color=COLORS[1])
axes[1].set_title('Top 12 Crops by Avg Yield (Quintal/Hectare)', fontweight='bold')
axes[1].set_xlabel('Yield (Quintal/Hectare)')

plt.tight_layout()
plt.savefig('outputs/01_top_crops_cost_yield.png', bbox_inches='tight')
plt.close()
print("  Saved: 01_top_crops_cost_yield.png")

# ── 3.3 State-wise average cost ─────────────
fig, ax = plt.subplots(figsize=(14, 6))
state_cost = (cost_df.groupby('State')['Cost_Cultivation_C2']
              .mean().sort_values(ascending=False).head(15))
sns.barplot(x=state_cost.values, y=state_cost.index, palette='Blues_r', ax=ax)
ax.set_title('Top 15 States by Avg Cultivation Cost (C2)', fontweight='bold')
ax.set_xlabel('Avg Cost (₹/Hectare)')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
plt.tight_layout()
plt.savefig('outputs/02_state_wise_cost.png', bbox_inches='tight')
plt.close()
print("  Saved: 02_state_wise_cost.png")

# ── 3.4 Correlation heatmap ──────────────────
fig, ax = plt.subplots(figsize=(8, 6))
corr = cost_df[['Cost_Cultivation_A2FL', 'Cost_Cultivation_C2',
                 'Cost_Production_C2', 'Yield_Quintal_Hectare']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            square=True, linewidths=0.5, ax=ax)
ax.set_title('Correlation Matrix — Cost & Yield Features', fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/03_correlation_heatmap.png', bbox_inches='tight')
plt.close()
print("  Saved: 03_correlation_heatmap.png")

# ── 3.5 Distribution plots ───────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
num_cols = ['Cost_Cultivation_A2FL', 'Cost_Cultivation_C2',
            'Cost_Production_C2', 'Yield_Quintal_Hectare']
labels   = ['Cost Cultivation A2+FL (₹/Ha)', 'Cost Cultivation C2 (₹/Ha)',
            'Cost Production C2 (₹/Quintal)', 'Yield (Quintal/Ha)']
for ax, col, lbl in zip(axes.flatten(), num_cols, labels):
    sns.histplot(cost_df[col], kde=True, ax=ax, color=COLORS[2])
    ax.set_title(f'Distribution: {lbl}', fontweight='bold')
    ax.set_xlabel(lbl)
plt.tight_layout()
plt.savefig('outputs/04_distributions.png', bbox_inches='tight')
plt.close()
print("  Saved: 04_distributions.png")

# ── 3.6 Production trend (macro) ─────────────
food_row = macro_df[macro_df['Particulars'].str.contains('Foodgrains$', na=False, regex=True)]
if not food_row.empty:
    years = [int(c.split('-')[1]) for c in year_cols]
    vals  = food_row[year_cols].values.flatten()
    mask  = ~np.isnan(vals.astype(float))
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(np.array(years)[mask], vals[mask].astype(float),
            marker='o', linewidth=2.5, color=COLORS[3])
    ax.fill_between(np.array(years)[mask], vals[mask].astype(float),
                    alpha=0.15, color=COLORS[3])
    ax.set_title('India Foodgrain Production Trend (1993–2014)', fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Production (Million Tons)')
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig('outputs/05_production_trend.png', bbox_inches='tight')
    plt.close()
    print("  Saved: 05_production_trend.png")

# ── 3.7 Production index trend ───────────────
fig, ax = plt.subplots(figsize=(13, 5))
year_idx_cols = [c for c in index_df.columns if c != 'Crop']
for i, (_, row) in enumerate(index_df.iterrows()):
    vals = row[year_idx_cols].values.astype(float)
    ax.plot(year_idx_cols, vals, marker='o', label=row['Crop'], color=COLORS[i % len(COLORS)])
ax.set_title('Crop Production Index Trend (2004–2012, Base 2004-05=100)', fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Production Index')
ax.legend(fontsize=8, loc='upper left')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('outputs/06_production_index_trend.png', bbox_inches='tight')
plt.close()
print("  Saved: 06_production_index_trend.png")

# ── 3.8 Boxplot: yield by crop ───────────────
top10_crops = cost_df['Crop'].value_counts().head(10).index
df_top10 = cost_df[cost_df['Crop'].isin(top10_crops)]
fig, ax = plt.subplots(figsize=(14, 6))
sns.boxplot(data=df_top10, x='Crop', y='Yield_Quintal_Hectare',
            palette='Set3', ax=ax)
ax.set_title('Yield Distribution by Crop (Top 10 Crops)', fontweight='bold')
ax.set_xlabel('Crop')
ax.set_ylabel('Yield (Quintal/Hectare)')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('outputs/07_yield_boxplot.png', bbox_inches='tight')
plt.close()
print("  Saved: 07_yield_boxplot.png")

# ── 3.9 Season distribution ──────────────────
season_counts = variety_df['Season'].value_counts().dropna().head(10)
fig, ax = plt.subplots(figsize=(10, 5))
season_counts.plot(kind='bar', color=COLORS[:len(season_counts)], ax=ax)
ax.set_title('Crop Variety Count by Season Type', fontweight='bold')
ax.set_xlabel('Season / Duration')
ax.set_ylabel('Count')
plt.xticks(rotation=40)
plt.tight_layout()
plt.savefig('outputs/08_season_distribution.png', bbox_inches='tight')
plt.close()
print("  Saved: 08_season_distribution.png")

# ── 3.10 Scatter: Cost vs Yield ──────────────
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(cost_df['Cost_Cultivation_C2'],
           cost_df['Yield_Quintal_Hectare'],
           alpha=0.6, color=COLORS[4], edgecolors='k', linewidths=0.4)
ax.set_title('Cultivation Cost C2 vs Yield', fontweight='bold')
ax.set_xlabel('Cost Cultivation C2 (₹/Hectare)')
ax.set_ylabel('Yield (Quintal/Hectare)')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
plt.tight_layout()
plt.savefig('outputs/09_cost_vs_yield_scatter.png', bbox_inches='tight')
plt.close()
print("  Saved: 09_cost_vs_yield_scatter.png")

# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[4] Feature Engineering ...")

df_model = cost_df.copy()

# Encode Crop and State
le_crop  = LabelEncoder()
le_state = LabelEncoder()
df_model['Crop_Enc']  = le_crop.fit_transform(df_model['Crop'])
df_model['State_Enc'] = le_state.fit_transform(df_model['State'])

# Derived features
df_model['Cost_Ratio']          = df_model['Cost_Cultivation_C2'] / (df_model['Cost_Cultivation_A2FL'] + 1)
df_model['Cost_Per_Yield_Unit'] = df_model['Cost_Cultivation_C2'] / (df_model['Yield_Quintal_Hectare'] + 1)
df_model['Log_Cost_C2']         = np.log1p(df_model['Cost_Cultivation_C2'])
df_model['Log_Yield']           = np.log1p(df_model['Yield_Quintal_Hectare'])

FEATURES = ['Crop_Enc', 'State_Enc',
            'Cost_Cultivation_A2FL', 'Cost_Cultivation_C2',
            'Cost_Ratio', 'Cost_Per_Yield_Unit']

TARGET_YIELD = 'Yield_Quintal_Hectare'
TARGET_COST  = 'Cost_Production_C2'

print(f"  Features used : {FEATURES}")
print(f"  Targets       : [{TARGET_YIELD}] and [{TARGET_COST}]")

# ─────────────────────────────────────────────
# 5. MODEL BUILDING — HELPER FUNCTIONS
# ─────────────────────────────────────────────

def evaluate_models(X_train, X_test, y_train, y_test, target_name):
    """Train multiple regressors, return results DataFrame + best model."""
    models = {
        'Linear Regression'       : LinearRegression(),
        'Ridge Regression'        : Ridge(alpha=1.0),
        'Decision Tree'           : DecisionTreeRegressor(max_depth=6, random_state=42),
        'Random Forest'           : RandomForestRegressor(n_estimators=100, random_state=42,
                                                          n_jobs=-1),
        'Gradient Boosting'       : GradientBoostingRegressor(n_estimators=100,
                                                               learning_rate=0.1,
                                                               max_depth=5, random_state=42),
    }

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    results = []
    trained = {}
    for name, model in models.items():
        model.fit(X_tr_s, y_train)
        y_pred = model.predict(X_te_s)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)
        cv   = cross_val_score(model, X_tr_s, y_train, cv=5,
                               scoring='r2').mean()
        results.append({'Model': name, 'RMSE': round(rmse, 3),
                        'MAE': round(mae, 3), 'R²': round(r2, 4),
                        'CV R² (mean)': round(cv, 4)})
        trained[name] = (model, scaler, y_pred)

    results_df = pd.DataFrame(results).sort_values('R²', ascending=False)
    best_name  = results_df.iloc[0]['Model']
    best_row   = results_df.iloc[0]
    print(f"\n  {target_name}")
    print(f"  Best model: {best_name} | R²={best_row['R²']:.4f} | RMSE={best_row['RMSE']:.3f} | MAE={best_row['MAE']:.3f}")
    return results_df, trained, best_name

def plot_model_comparison(results_df, target_name, fname):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ['RMSE', 'MAE', 'R²']
    palette = sns.color_palette('Set2', len(results_df))
    for ax, metric in zip(axes, metrics):
        bars = ax.barh(results_df['Model'], results_df[metric], color=palette)
        ax.set_title(f'{metric}', fontweight='bold')
        ax.set_xlabel(metric)
        for bar, val in zip(bars, results_df[metric]):
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=9)
    fig.suptitle(f'Model Comparison — {target_name}', fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'outputs/{fname}', bbox_inches='tight')
    plt.close()

def plot_actual_vs_pred(y_test, y_pred, model_name, target_name, fname):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Scatter
    axes[0].scatter(y_test, y_pred, alpha=0.6, color=COLORS[0],
                    edgecolors='k', linewidths=0.3)
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    axes[0].plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='Perfect Fit')
    axes[0].set_xlabel('Actual'); axes[0].set_ylabel('Predicted')
    axes[0].set_title(f'Actual vs Predicted — {model_name}', fontweight='bold')
    axes[0].legend()
    # Residuals
    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6, color=COLORS[1],
                    edgecolors='k', linewidths=0.3)
    axes[1].axhline(0, color='red', linestyle='--', linewidth=1.5)
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Residual')
    axes[1].set_title('Residual Plot', fontweight='bold')
    fig.suptitle(f'{target_name} — {model_name}', fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'outputs/{fname}', bbox_inches='tight')
    plt.close()

def plot_feature_importance(model, features, title, fname):
    if hasattr(model, 'feature_importances_'):
        imp = pd.Series(model.feature_importances_, index=features).sort_values()
        fig, ax = plt.subplots(figsize=(9, 5))
        imp.plot(kind='barh', color=COLORS[5], ax=ax)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Importance')
        plt.tight_layout()
        plt.savefig(f'outputs/{fname}', bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fname}")

# ─────────────────────────────────────────────
# 6. TARGET 1: PREDICT CROP PRODUCTION YIELD
# ─────────────────────────────────────────────
print("\n[5] ML — Target 1: Predicting Crop Yield (Quintal/Hectare) ...")

X = df_model[FEATURES]
y_yield = df_model[TARGET_YIELD]

X_tr, X_te, y_tr, y_te = train_test_split(X, y_yield, test_size=0.2,
                                            random_state=42)
res_yield, trained_yield, best_yield = evaluate_models(
    X_tr, X_te, y_tr, y_te, 'Crop Yield Prediction')

plot_model_comparison(res_yield, 'Crop Yield (Quintal/Hectare)',
                      '10_yield_model_comparison.png')
print("  Saved: 10_yield_model_comparison.png")

best_model_yield, best_scaler_yield, best_pred_yield = trained_yield[best_yield]
plot_actual_vs_pred(y_te.values, best_pred_yield, best_yield,
                    'Crop Yield Prediction', '11_yield_actual_vs_pred.png')
print("  Saved: 11_yield_actual_vs_pred.png")

# ─────────────────────────────────────────────
# 7. TARGET 2: PREDICT CULTIVATION COST
# ─────────────────────────────────────────────
print("\n[6] ML — Target 2: Predicting Cost of Production (₹/Quintal) ...")

FEATURES_COST = ['Crop_Enc', 'State_Enc',
                  'Cost_Cultivation_A2FL', 'Yield_Quintal_Hectare',
                  'Cost_Ratio']

y_cost = df_model[TARGET_COST]
X_cost = df_model[FEATURES_COST]

Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X_cost, y_cost,
                                                 test_size=0.2, random_state=42)
res_cost, trained_cost, best_cost = evaluate_models(
    Xc_tr, Xc_te, yc_tr, yc_te, 'Cultivation Cost Prediction')

plot_model_comparison(res_cost, 'Cost of Production (₹/Quintal)',
                      '13_cost_model_comparison.png')
print("  Saved: 13_cost_model_comparison.png")

best_model_cost, best_scaler_cost, best_pred_cost = trained_cost[best_cost]
plot_actual_vs_pred(yc_te.values, best_pred_cost, best_cost,
                    'Cost of Production Prediction', '14_cost_actual_vs_pred.png')
print("  Saved: 14_cost_actual_vs_pred.png")

# ─────────────────────────────────────────────
# 8. PREDICTION SUMMARY DASHBOARD
# ─────────────────────────────────────────────
print("\n[7] Generating Final Summary Dashboard ...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ── Actual vs Predicted: Yield ────────────────
ax = axes[0, 0]
ax.scatter(y_te.values, best_pred_yield, alpha=0.6,
           color=COLORS[0], edgecolors='k', linewidths=0.3, s=50)
mn, mx = min(y_te.min(), best_pred_yield.min()), max(y_te.max(), best_pred_yield.max())
ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2)
ax.set_title(f'Yield Prediction — {best_yield}', fontweight='bold')
ax.set_xlabel('Actual Yield (Q/Ha)')
ax.set_ylabel('Predicted Yield (Q/Ha)')
r2_y = r2_score(y_te, best_pred_yield)
ax.text(0.05, 0.93, f'R² = {r2_y:.4f}', transform=ax.transAxes,
        fontsize=11, color='darkgreen', fontweight='bold')

# ── Actual vs Predicted: Cost ─────────────────
ax = axes[0, 1]
ax.scatter(yc_te.values, best_pred_cost, alpha=0.6,
           color=COLORS[1], edgecolors='k', linewidths=0.3, s=50)
mn, mx = min(yc_te.min(), best_pred_cost.min()), max(yc_te.max(), best_pred_cost.max())
ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2)
ax.set_title(f'Cost Prediction — {best_cost}', fontweight='bold')
ax.set_xlabel('Actual Cost (₹/Quintal)')
ax.set_ylabel('Predicted Cost (₹/Quintal)')
r2_c = r2_score(yc_te, best_pred_cost)
ax.text(0.05, 0.93, f'R² = {r2_c:.4f}', transform=ax.transAxes,
        fontsize=11, color='darkgreen', fontweight='bold')

# ── Model R² Comparison: Yield ────────────────
ax = axes[1, 0]
palette = sns.color_palette('Set2', len(res_yield))
bars = ax.bar(res_yield['Model'], res_yield['R²'], color=palette)
ax.set_title('R² Score Comparison — Yield Models', fontweight='bold')
ax.set_ylabel('R² Score')
ax.set_ylim(0, 1.05)
plt.setp(ax.get_xticklabels(), rotation=20, ha='right', fontsize=9)
for bar, val in zip(bars, res_yield['R²']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', fontsize=9)

# ── Model R² Comparison: Cost ─────────────────
ax = axes[1, 1]
palette2 = sns.color_palette('Set1', len(res_cost))
bars2 = ax.bar(res_cost['Model'], res_cost['R²'], color=palette2)
ax.set_title('R² Score Comparison — Cost Models', fontweight='bold')
ax.set_ylabel('R² Score')
ax.set_ylim(0, 1.05)
plt.setp(ax.get_xticklabels(), rotation=20, ha='right', fontsize=9)
for bar, val in zip(bars2, res_cost['R²']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', fontsize=9)

fig.suptitle('Agriculture Crop Production Prediction — Summary Dashboard',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('outputs/16_summary_dashboard.png', bbox_inches='tight')
plt.show()
plt.close()
print("  Saved: 16_summary_dashboard.png")

# ─────────────────────────────────────────────
# 9. FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL RESULTS")
print("=" * 60)

yr = res_yield[res_yield['Model'] == best_yield].iloc[0]
cr = res_cost[res_cost['Model'] == best_cost].iloc[0]

print(f"  Yield prediction is best with: {best_yield}")
print(f"    • R² = {yr['R²']:.4f}")
print(f"    • RMSE = {yr['RMSE']:.3f}")
print(f"    • MAE = {yr['MAE']:.3f}")
print(f"  Cost prediction is best with: {best_cost}")
print(f"    • R² = {cr['R²']:.4f}")
print(f"    • RMSE = {cr['RMSE']:.3f}")
print(f"    • MAE = {cr['MAE']:.3f}")
print("\n  The summary charts and plots are saved in the outputs folder.")
print("=" * 60)
print("  ✅ All done — thanks for running the script")
print("=" * 60)
