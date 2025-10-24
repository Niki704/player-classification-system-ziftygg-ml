"""
Model Training Script for Zifty Player Classification
Trains multiple ML models and selects the best one
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ZIFTY PLAYER CLASSIFICATION - MODEL TRAINING")
print("="*70)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[1/8] Loading processed data...")

try:
    data = pd.read_csv('data/processed/zifty_player_data_complete_2000.csv')
    print(f"Successfully loaded {len(data)} records")
except FileNotFoundError:
    print("ERROR: Could not find 'data/processed/zifty_player_data_complete_2000.csv'")
    print("Please run the data generation script first.")
    exit()

print(f"   Features: {data.shape[1]} columns")
print(f"   Samples: {data.shape[0]} rows")

# ============================================================================
# STEP 2: PREPARE FEATURES AND TARGET
# ============================================================================
print("\n[2/8] Preparing features and target variable...")

# Select features for the model
feature_columns = [
    'mp_kd_ratio',
    'mp_legendary_streak', 
    'experience_level',
    'daily_play_time',
    'codm_experience'
]

# Define numerical and categorical features
numerical_features = ['mp_kd_ratio', 'mp_legendary_streak', 'experience_level']
categorical_features = ['daily_play_time', 'codm_experience']

# Prepare X (features) and y (target)
X = data[feature_columns].copy()
y = data['performance_score'].copy()
y_class = data['player_class'].copy()

print(f"   Features prepared:")
print(f"   Numerical: {numerical_features}")
print(f"   Categorical: {categorical_features}")
print(f"   Target: performance_score (regression)")
print(f"   Target classes: {y_class.unique()}")

# ============================================================================
# STEP 3: TRAIN-TEST SPLIT
# ============================================================================
print("\n[3/8] Splitting data into training and testing sets...")

# Check class distribution first
print("\nClass distribution:")
print(y_class.value_counts().sort_index())

# Check if any class has fewer than 2 samples
min_class_count = y_class.value_counts().min()

if min_class_count < 2:
    print(f"\nWarning: Smallest class has only {min_class_count} sample(s)")
    print("Using random split without stratification")
    
    X_train, X_test, y_train, y_test, y_class_train, y_class_test = train_test_split(
        X, y, y_class, test_size=0.2, random_state=42
    )
else:
    print("Using stratified split to maintain class proportions")
    
    X_train, X_test, y_train, y_test, y_class_train, y_class_test = train_test_split(
        X, y, y_class, test_size=0.2, random_state=42, stratify=y_class
    )

print(f"   Data split complete:")
print(f"   Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Testing set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

# ============================================================================
# STEP 4: CREATE PREPROCESSING PIPELINE
# ============================================================================
print("\n[4/8] Creating preprocessing pipeline...")

# Create preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ],
    remainder='drop'
)

print("  Preprocessing pipeline created")
print("   - Numerical features: StandardScaler")
print("   - Categorical features: OneHotEncoder")

# ============================================================================
# STEP 5: TRAIN MULTIPLE MODELS
# ============================================================================
print("\n[5/8] Training multiple regression models...")

# Define models to train
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
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
}

# Store results
results = {}
trained_models = {}

print("\nTraining models (this may take a minute)...\n")

for name, model in models.items():
    print(f"Training {name}...")
    
    # Create pipeline with preprocessing and model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Cross-validation score
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, 
                                scoring='neg_mean_squared_error', n_jobs=-1)
    cv_mse = -cv_scores.mean()
    
    # Store results
    results[name] = {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'cv_mse': cv_mse,
        'predictions': y_pred_test
    }
    
    trained_models[name] = pipeline
    
    print(f"   Test MSE: {test_mse:.2f}, Test R2: {test_r2:.3f}, Test MAE: {test_mae:.2f}")

# ============================================================================
# STEP 6: SELECT BEST MODEL
# ============================================================================
print("\n[6/8] Evaluating and selecting best model...")

# Find best model based on test MSE
best_model_name = min(results, key=lambda k: results[k]['test_mse'])
best_model = trained_models[best_model_name]

print(f"\n Best Model: {best_model_name}")
print(f"   Test MSE: {results[best_model_name]['test_mse']:.2f}")
print(f"   Test MAE: {results[best_model_name]['test_mae']:.2f}")
print(f"   Test R2: {results[best_model_name]['test_r2']:.3f}")
print(f"   CV MSE: {results[best_model_name]['cv_mse']:.2f}")

# Print comparison table
print("\n  Model Comparison:")
print("-" * 80)
print(f"{'Model':<20} {'Train MSE':<12} {'Test MSE':<12} {'Test R2':<12} {'Test MAE':<12}")
print("-" * 80)
for name, metrics in results.items():
    marker = "*" if name == best_model_name else "  "
    print(f"{marker} {name:<18} {metrics['train_mse']:<12.2f} {metrics['test_mse']:<12.2f} "
          f"{metrics['test_r2']:<12.3f} {metrics['test_mae']:<12.2f}")
print("-" * 80)

# ============================================================================
# STEP 7: EVALUATE CLASSIFICATION ACCURACY
# ============================================================================
print("\n[7/8] Evaluating classification accuracy...")

# Get predictions from best model
y_pred_test = best_model.predict(X_test)

# Convert predictions to classes
def assign_class(score):
    if score >= 71: return 'A'
    elif score >= 51: return 'B'
    elif score >= 35: return 'C'
    elif score >= 26: return 'D'
    else: return 'E'

y_pred_class = [assign_class(score) for score in y_pred_test]

# Calculate classification accuracy
class_accuracy = accuracy_score(y_class_test, y_pred_class)

print(f"\n Classification Accuracy: {class_accuracy*100:.2f}%")

# Classification report
print("\n Classification Report:")
print(classification_report(y_class_test, y_pred_class, 
                          labels=['A', 'B', 'C', 'D', 'E'],
                          zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_class_test, y_pred_class, labels=['A', 'B', 'C', 'D', 'E'])

# ============================================================================
# STEP 8: SAVE MODEL AND GENERATE REPORTS
# ============================================================================
print("\n[8/8] Saving model and generating reports...")

# Save the best model
model_filename = 'models/player_classification_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(best_model, f)
print(f" Model saved: {model_filename}")

# Save model metrics
metrics_filename = 'models/model_metrics.json'
metrics_to_save = {
    'best_model': best_model_name,
    'test_mse': float(results[best_model_name]['test_mse']),
    'test_mae': float(results[best_model_name]['test_mae']),
    'test_r2': float(results[best_model_name]['test_r2']),
    'classification_accuracy': float(class_accuracy),
    'all_models': {
        name: {
            'test_mse': float(metrics['test_mse']),
            'test_r2': float(metrics['test_r2']),
            'test_mae': float(metrics['test_mae'])
        }
        for name, metrics in results.items()
    }
}

with open(metrics_filename, 'w') as f:
    json.dump(metrics_to_save, f, indent=4)
print(f" Metrics saved: {metrics_filename}")

# Create visualizations
print("\n Creating performance visualizations...")

fig = plt.figure(figsize=(16, 10))

# 1. Actual vs Predicted (Regression)
plt.subplot(2, 3, 1)
plt.scatter(y_test, y_pred_test, alpha=0.5, s=30)
plt.plot([0, 100], [0, 100], 'r--', linewidth=2)
plt.xlabel('Actual Performance Score')
plt.ylabel('Predicted Performance Score')
plt.title(f'{best_model_name}: Actual vs Predicted', fontweight='bold')
plt.grid(alpha=0.3)

# 2. Residual Plot
plt.subplot(2, 3, 2)
residuals = y_test - y_pred_test
plt.scatter(y_pred_test, residuals, alpha=0.5, s=30)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Predicted Performance Score')
plt.ylabel('Residuals')
plt.title('Residual Plot', fontweight='bold')
plt.grid(alpha=0.3)

# 3. Confusion Matrix
plt.subplot(2, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['A', 'B', 'C', 'D', 'E'],
            yticklabels=['A', 'B', 'C', 'D', 'E'])
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.title(f'Confusion Matrix (Accuracy: {class_accuracy*100:.1f}%)', fontweight='bold')

# 4. Model Comparison
plt.subplot(2, 3, 4)
model_names = list(results.keys())
test_mse_values = [results[name]['test_mse'] for name in model_names]
colors = ['#2E86AB' if name == best_model_name else '#A9A9A9' for name in model_names]
bars = plt.bar(range(len(model_names)), test_mse_values, color=colors)
plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
plt.ylabel('Test MSE')
plt.title('Model Performance Comparison', fontweight='bold')
plt.grid(axis='y', alpha=0.3)

# 5. R² Score Comparison
plt.subplot(2, 3, 5)
r2_values = [results[name]['test_r2'] for name in model_names]
colors = ['#2E86AB' if name == best_model_name else '#A9A9A9' for name in model_names]
bars = plt.bar(range(len(model_names)), r2_values, color=colors)
plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
plt.ylabel('R² Score')
plt.title('R² Score Comparison', fontweight='bold')
plt.ylim([0, 1])
plt.grid(axis='y', alpha=0.3)

# 6. Prediction Distribution
plt.subplot(2, 3, 6)
plt.hist(y_test, bins=20, alpha=0.5, label='Actual', color='blue', edgecolor='black')
plt.hist(y_pred_test, bins=20, alpha=0.5, label='Predicted', color='orange', edgecolor='black')
plt.xlabel('Performance Score')
plt.ylabel('Frequency')
plt.title('Distribution: Actual vs Predicted', fontweight='bold')
plt.legend()

plt.tight_layout()
plt.savefig('reports/model_performance_report.png', dpi=300, bbox_inches='tight')
print(f" Saved: reports/model_performance_report.png")

# Create confusion matrix as separate detailed plot
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'},
            xticklabels=['A', 'B', 'C', 'D', 'E'],
            yticklabels=['A', 'B', 'C', 'D', 'E'],
            linewidths=0.5, linecolor='gray')
plt.xlabel('Predicted Class', fontsize=12)
plt.ylabel('Actual Class', fontsize=12)
plt.title(f'Player Class Prediction Confusion Matrix\nAccuracy: {class_accuracy*100:.2f}%', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f" Saved: reports/confusion_matrix.png")

# Feature importance (if Random Forest or Gradient Boosting)
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    print("\n Extracting feature importance...")
    
    # Get the regressor from pipeline
    regressor = best_model.named_steps['regressor']
    
    # Get feature names after preprocessing
    preprocessor_fitted = best_model.named_steps['preprocessor']
    
    # Get feature names
    cat_features = list(preprocessor_fitted.named_transformers_['cat'].get_feature_names_out(categorical_features))
    all_features = numerical_features + cat_features
    
    # Get importances
    importances = regressor.feature_importances_
    
    # Create dataframe
    feature_importance_df = pd.DataFrame({
        'feature': all_features,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    top_features = feature_importance_df.head(10)
    plt.barh(range(len(top_features)), top_features['importance'], color='#2E86AB')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importances', fontweight='bold', fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('reports/feature_importance.png', dpi=300, bbox_inches='tight')
    print(f" Saved: reports/feature_importance.png")
    
    print("\nTop 5 Most Important Features:")
    for idx, row in feature_importance_df.head(5).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")

print("\n" + "="*70)
print("MODEL TRAINING COMPLETE!")
print("="*70)
print(f"\n Generated files:")
print(f"   1. {model_filename}")
print(f"   2. {metrics_filename}")
print(f"   3. reports/model_performance_report.png")
print(f"   4. reports/confusion_matrix.png")
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    print(f"   5. reports/feature_importance.png")

print(f"\n Model Performance Summary:")
print(f"   Best Model: {best_model_name}")
print(f"   Test R2 Score: {results[best_model_name]['test_r2']:.3f}")
print(f"   Test MAE: {results[best_model_name]['test_mae']:.2f} points")
print(f"   Classification Accuracy: {class_accuracy*100:.2f}%")