"""
Model Training Script for Zifty Player Classification
Trains ML models and securely exports them to TypeScript
"""

import pandas as pd
import numpy as np
import pickle
import json
import m2cgen as m2c
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ZIFTY PLAYER CLASSIFICATION - MODEL TRAINING")
print("="*70)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[1/7] Loading processed data...")

try:
    data = pd.read_csv('data/processed/zifty_player_data_complete_105.csv')
    print(f"Successfully loaded {len(data)} records")
except FileNotFoundError:
    print("ERROR: Could not find 'data/processed/zifty_player_data_complete_105.csv'")
    exit()

# ============================================================================
# STEP 2: PREPARE FEATURES AND TARGET
# ============================================================================
print("\n[2/7] Preparing features and target variable...")

play_time_map = {
    'Less than 1 hour': 0, '1-2 hours': 1, '2-3 hours': 2, 'More than 3 hours': 3
}
exp_map = {
    'Less than 1 year': 0, '1-2 years': 1, '2-3 years': 2, 'More than 3 years': 3
}

data['daily_play_time'] = data['daily_play_time'].map(play_time_map)
data['codm_experience'] = data['codm_experience'].map(exp_map)

feature_columns = [
    'mp_kd_ratio',
    'mp_legendary_streak', 
    'experience_level',
    'daily_play_time',
    'codm_experience'
]

X = data[feature_columns].copy()
y = data['performance_score'].copy()
y_class = data['player_class'].copy()

# ============================================================================
# STEP 3: TRAIN-TEST SPLIT
# ============================================================================
print("\n[3/7] Splitting data into training and testing sets...")

min_class_count = y_class.value_counts().min()
if min_class_count < 2:
    X_train, X_test, y_train, y_test, y_class_train, y_class_test = train_test_split(
        X, y, y_class, test_size=0.2, random_state=42
    )
else:
    X_train, X_test, y_train, y_test, y_class_train, y_class_test = train_test_split(
        X, y, y_class, test_size=0.2, random_state=42, stratify=y_class
    )

print(f"   Training set: {len(X_train)} samples")
print(f"   Testing set: {len(X_test)} samples")

# ============================================================================
# STEP 4: TRAIN MODELS (NATIVE ALGORITHMS ONLY)
# ============================================================================
print("\n[4/7] Training regression models...")

# Only using m2cgen-compatible models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
}

results = {}
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    
    results[name] = {
        'test_mse': test_mse,
        'test_mae': test_mae,
        'test_r2': test_r2,
    }
    trained_models[name] = model 

# ============================================================================
# STEP 5: SELECT BEST MODEL & EVALUATE
# ============================================================================
print("\n[5/7] Evaluating and selecting best model...")

best_model_name = min(results, key=lambda k: results[k]['test_mse'])
best_model = trained_models[best_model_name]

print(f"\n Best Model: {best_model_name}")
print(f"   Test MSE: {results[best_model_name]['test_mse']:.2f}")
print(f"   Test R2: {results[best_model_name]['test_r2']:.3f}")

y_pred_test = best_model.predict(X_test)

def assign_class(score):
    if score >= 75: return 'A'
    elif score >= 55: return 'B'
    elif score >= 35: return 'C'
    elif score >= 20: return 'D'
    else: return 'E'

y_pred_class = [assign_class(score) for score in y_pred_test]
class_accuracy = accuracy_score(y_class_test, y_pred_class)

print(f"\n Classification Accuracy: {class_accuracy*100:.2f}%")

if best_model_name == 'Random Forest':
    print("\n Top Feature Importances:")
    importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance_df.iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")

# ============================================================================
# STEP 6: SAVE PYTHON BACKUPS
# ============================================================================
print("\n[6/7] Saving Python model backups...")

model_filename = 'models/player_classification_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(best_model, f)

metrics_filename = 'models/model_metrics.json'
metrics_to_save = {
    'best_model': best_model_name,
    'classification_accuracy': float(class_accuracy),
    'test_r2': float(results[best_model_name]['test_r2'])
}
with open(metrics_filename, 'w') as f:
    json.dump(metrics_to_save, f, indent=4)

# ============================================================================
# STEP 7: EXPORT NATIVE TYPESCRIPT MODEL FOR ZIFTY GG
# ============================================================================
print("\n[7/7] Transpiling model to TypeScript for complete independence...")

try:
    js_code = m2c.export_to_javascript(best_model)
    
    ts_code = f"""// AUTO-GENERATED ML MODEL
// Transpiled from Scikit-Learn: {best_model_name}
// This allows Zifty GG to grade players natively without a Python backend.

export function predictScore(input: number[]): number {{
{js_code.replace('function score(input) {', '').rsplit('}', 1)[0]}
}}
"""
    
    nextjs_model_path = '../zifty-gg/src/lib/ml/model.ts' 
    os.makedirs(os.path.dirname(nextjs_model_path), exist_ok=True)
    
    with open(nextjs_model_path, 'w') as f:
        f.write(ts_code)
        
    print(f" Successfully exported pure TypeScript model to: {nextjs_model_path}")
    
except Exception as e:
    print(f" ERROR: Could not export model to TypeScript. {e}")

print("\n" + "="*70)
print("PIPELINE COMPLETE!")
print("="*70)