# Importing libraries

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
import joblib

# Setup directory
output_dir = r'path'
os.makedirs(output_dir, exist_ok=True)

# Load dataset
X = pd.read_csv(r'path', header=None)  
labels_df = pd.read_csv(r'path', header=None)  
y = pd.read_csv(r'path', header=None)  

print(f"X shape: {X.shape}, y shape: {y.shape}, labels_df shape: {labels_df.shape}")

X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(
    X, y, labels_df, test_size=0.2, random_state=42, stratify=y
)

# Define the LGBM parameters
# Baseline
params = {
    'objective': 'binary',
    'importance_type': 'gain',
    'n_jobs': -1,
}

# Or

#Hyperparameter tunned
params = {
    'num_leaves': 31,
    'n_estimators': 500,
    'min_data_in_leaf': 50,
    'max_depth': 10,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_freq': 10,
    'bagging_fraction': 0.9,
    'objective': 'binary',
    'n_jobs': -1,
    'importance_type': 'gain',  
    'random_state': 42          
}

# Initialize the machine learning classifier
model = lgb.LGBMClassifier(**params)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Function to calculate feature importance ratio for adaptive step size
def compute_feature_importance_ratio(model):
    # Train the model on a subset to calculate feature importance
    model.fit(X_train, y_train)
    feature_importances = model.feature_importances_
    median_importance = np.median(feature_importances)
    important_features_ratio = np.sum(feature_importances > median_importance) / len(feature_importances)
    return important_features_ratio

# Modified nonlinear step scaling function
def nonlinear_step_scaling(n_samples, base_step=0.1, decay_rate=0.5, n_total=None):
    if n_total is None:
        n_total = n_samples

    t1 = 0.9 * n_total
    t2 = 0.8 * n_total
    t3 = 0.67 * n_total
    t4 = 0.5 * n_total
    t5 = 0.33 * n_total
    t6 = 0.25 * n_total
    t7 = 0.17 * n_total
    t8 = 0.1 * n_total
    t9 = 0.05 * n_total

    if n_samples > t1:
        return base_step * np.exp(-decay_rate * (n_samples - t1) / n_total)
    elif n_samples > t2:
        return base_step * np.exp(-decay_rate * (n_samples - t2) / n_total) * 0.9
    elif n_samples > t3:
        return base_step * np.exp(-decay_rate * (n_samples - t3) / n_total) * 0.8
    elif n_samples > t4:
        return base_step * np.exp(-decay_rate * (n_samples - t4) / n_total) * 0.7
    elif n_samples > t5:
        return base_step * np.exp(-decay_rate * (n_samples - t5) / n_total) * 0.6
    elif n_samples > t6:
        return base_step * np.exp(-decay_rate * (n_samples - t6) / n_total) * 0.5
    elif n_samples > t7:
        return base_step * np.exp(-decay_rate * (n_samples - t7) / n_total) * 0.4
    elif n_samples > t8:
        return base_step * np.exp(-decay_rate * (n_samples - t8) / n_total) * 0.3
    elif n_samples > t9:
        return base_step * np.exp(-decay_rate * (n_samples - t9) / n_total) * 0.25
    else:
        return base_step * np.exp(-decay_rate * (n_samples - 1) / n_total) * 0.2

# Determine the optimal step size based on training data size and feature importance
base_reduction_factor = 0.1  
decay_rate = 0.5  
n_samples = len(X_train)
nonlinear_step = nonlinear_step_scaling(n_samples, base_step=base_reduction_factor, decay_rate=decay_rate)

# Incorporate feature importance ratio into the step size calculation
important_features_ratio = compute_feature_importance_ratio(model)
adaptive_step_size = max(0.01, nonlinear_step * important_features_ratio)

print("nonlinear_step =", nonlinear_step)
print("importance_ratio =", important_features_ratio)
print("adaptive_step_size =", adaptive_step_size)
print("~features removed/iter =", int(np.ceil(adaptive_step_size * X_train.shape[1])))

rfecv = RFECV(
    estimator=model,
    step=adaptive_step_size,  
    cv=cv,
    scoring='f1',  
    n_jobs=-1,
    verbose=2
)

rfecv.fit(X_train, y_train)

# Get the optimal sample indices
optimal_indices = np.where(rfecv.support_)[0]

y_train = pd.Series(y_train).reset_index(drop=True)
labels_train = pd.DataFrame(labels_train).reset_index(drop=True)

# Extract optimal samples and labels based on indices
X_train_selected = X_train.iloc[optimal_indices, :].reset_index(drop=True)
y_train_selected = y_train.iloc[optimal_indices].reset_index(drop=True)
optimal_labels = labels_train.iloc[optimal_indices].values

print(f"Number of optimal samples: {len(optimal_indices)}")
print("Optimal sample indices and corresponding labels:")
for idx, label in zip(optimal_indices, optimal_labels):
    print(f"Sample Index: {idx}, Label: {label[0]}")

pd.DataFrame(optimal_indices, columns=['Sample Index']).to_csv(
    os.path.join(output_dir, 'optimal_sample_indices.csv'), index=False
)
pd.DataFrame(optimal_labels).to_csv(
    os.path.join(output_dir, 'optimal_sample_labels.csv'), index=False, header=False
)

joblib.dump(rfecv, os.path.join(output_dir, 'rfecv_model.joblib'))

# Train the model on the selected optimal samples in the training set
model.fit(X_train_selected, y_train_selected)

# Final prediction on the test set
y_pred = model.predict(X_test)

# Classification report
report = classification_report(y_test, y_pred, digits=4)
print("Classification Report on Test Set:")
with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
    f.write(report)

mean_test_scores = rfecv.cv_results_['mean_test_score']
std_test_scores = rfecv.cv_results_['std_test_score']

# Save cross-validation scores for each fold
cv_scores = [rfecv.cv_results_[f'split{i}_test_score'] for i in range(cv.n_splits)]
np.save(os.path.join(output_dir, 'cv_scores.npy'), cv_scores)

joblib.dump(model, os.path.join(output_dir, 'lgbm_model_optimal_samples.pkl'))
print(f"Optimal feature selection completed. Best f1: {max(mean_test_scores)} with {len(optimal_indices)} samples.")