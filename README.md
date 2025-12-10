# Classifying Decision-Based Effector Modalities from Fragmented iEEG via Dynamic Recursive Feature Elimination

This GitHub repo comprises the dynamic recursive feature elimination with cross-validation (DyRFECV) scripts employed on a Light Gradient Boosting Machine (LGBM) classifier with both baseline and hypereparameter tunned versions.

## DyRFECV framework
Python code trains and tests the DyRFECV framework on the LGBM classifer in two different settings (baseline and hypereparameter tunned) on the preprocessed neural data and reports performance.
dyrfecv.py (main script)
a.	Input: Dataset and splitting into train/test sets
b.	Initialise ML classifier: With/without hyperparameter tuning
c.  Model training and importance: Set up training strategy, fit on training data, and compute feature importance
d.	Adaptive RFE loop: Set up adaptive step size from training size and importance ratio, and remove the least informative samples
e.  Test set evaluation: Retrain on optimal sample subset, evaluate on held-out test set and Report F1
f.  Output: Optimal sample indices, save trained model, and metrics

### Usage
python dyrfecv.py (main script)   

### Licence
Released for peer review only.
Please do not redistribute or use commercially without permission.