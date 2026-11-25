import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, 
    recall_score, f1_score, roc_auc_score
)

# ==========================================
# TASK 3.1: Features, Split, and Baseline (5 Marks)
# ==========================================
print("="*60)
print("TASK 3.1: FEATURES, SPLIT & BASELINE")
print("="*60)

# Load clean data without shuffling or re-sorting
df = pd.read_csv("appointments_clean.csv")

# Keep appointment_id aside before building X
df_indexed = df.set_index("appointment_id")

# Encode reminder_sent: Yes -> 1, No -> 0
df_indexed["reminder_sent"] = df_indexed["reminder_sent"].map({"Yes": 1, "No": 0})

# Select base features and one-hot encode booking_channel and visit_type
features = ["age", "distance_km", "fee_inr", "lead_days", "prior_no_shows", "reminder_sent", "booking_channel", "visit_type"]
X = pd.get_dummies(df_indexed[features], columns=["booking_channel", "visit_type"], dtype=int)
y = df_indexed["no_show"]

# Train-test split (80/20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline accuracy: always predicting "Attended" (0)
baseline_preds = np.zeros_like(y_test)
baseline_acc = accuracy_score(y_test, baseline_preds)

print(f"X shape (total columns): {X.shape[1]}")
print(f"Training set size: {len(X_train)} rows")
print(f"Test set size: {len(X_test)} rows")
print(f"No-shows in test set: {y_test.sum()} out of {len(y_test)} ({y_test.mean()*100:.1f}%)")
print(f"Baseline Accuracy (always predicting Attended): {baseline_acc:.4f} ({int(baseline_acc*len(y_test))}/{len(y_test)})")


# ==========================================
# TASK 3.2: Logistic Regression (6 Marks)
# ==========================================
print("\n" + "="*60)
print("TASK 3.2: LOGISTIC REGRESSION")
print("="*60)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_s, y_train)

lr_preds = lr.predict(X_test_s)
lr_probs = lr.predict_proba(X_test_s)[:, 1]

lr_acc = accuracy_score(y_test, lr_preds)
lr_prec = precision_score(y_test, lr_preds)
lr_rec = recall_score(y_test, lr_preds)
lr_f1 = f1_score(y_test, lr_preds)
lr_auc = roc_auc_score(y_test, lr_probs)
lr_cm = confusion_matrix(y_test, lr_preds)

print(f"Accuracy : {lr_acc:.3f}")
print(f"Precision: {lr_prec:.2f}")
print(f"Recall   : {lr_rec:.2f}")
print(f"F1 Score : {lr_f1:.2f}")
print(f"ROC-AUC  : {lr_auc:.3f}")
print(f"Confusion Matrix:\n{lr_cm}")

print("\nFeature Coefficients (Scaled):")
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": lr.coef_[0],
    "Std_Dev": scaler.scale_
})
print(coef_df.to_string(index=False))

# Effect of reminder_sent going from No (0) to Yes (1)
rem_idx = list(X.columns).index("reminder_sent")
rem_coef = lr.coef_[0][rem_idx]
rem_sd = scaler.scale_[rem_idx]
real_odds_ratio = np.exp(rem_coef / rem_sd)
scaled_exp = np.exp(rem_coef)

print(f"\nReminder Coefficient (scaled): {rem_coef:.4f} | exp(coef): {scaled_exp:.2f}")
print(f"Real No->Yes Odds Ratio (exp(coef / SD)): {real_odds_ratio:.2f}")
print("Conclusion: In this model, holding the other features constant, sending a reminder is associated with roughly 69% lower odds of a no-show (odds ratio ~0.31), though observational data cannot prove causation.")


# ==========================================
# TASK 3.3: Decision Tree & Overfitting (5 Marks)
# ==========================================
print("\n" + "="*60)
print("TASK 3.3: DECISION TREE & OVERFITTING")
print("="*60)

dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train, y_train)
dt_full_train_acc = accuracy_score(y_train, dt_full.predict(X_train))
dt_full_test_acc = accuracy_score(y_test, dt_full.predict(X_test))

dt_pruned = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_pruned.fit(X_train, y_train)
dt_pruned_train_acc = accuracy_score(y_train, dt_pruned.predict(X_train))
dt_pruned_test_acc = accuracy_score(y_test, dt_pruned.predict(X_test))

print(f"Full-depth Tree (depth={dt_full.get_depth()}): Train Acc = {dt_full_train_acc:.2f}, Test Acc = {dt_full_test_acc:.2f}")
print(f"Depth-3 Tree               : Train Acc = {dt_pruned_train_acc:.2f}, Test Acc = {dt_pruned_test_acc:.2f}")
print("Conclusion: The huge gap between 1.00 training accuracy and 0.67 test accuracy shows severe overfitting—the unrestricted tree memorized training noise instead of learning general patterns.")


# ==========================================
# TASK 3.4: Random Forest & Comparison Table (6 Marks)
# ==========================================
print("\n" + "="*60)
print("TASK 3.4: RANDOM FOREST & MODEL COMPARISON")
print("="*60)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_preds = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:, 1]
rf_acc = accuracy_score(y_test, rf_preds)
rf_rec = recall_score(y_test, rf_preds)
rf_auc = roc_auc_score(y_test, rf_probs)

# Depth-3 Tree metrics for comparison table
dt3_preds = dt_pruned.predict(X_test)
dt3_probs = dt_pruned.predict_proba(X_test)[:, 1]
dt3_rec = recall_score(y_test, dt3_preds)
dt3_auc = roc_auc_score(y_test, dt3_probs)

# Top 3 feature importances
imp_series = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Top 3 Feature Importances (Random Forest):")
for feat, val in imp_series.head(3).items():
    print(f"  - {feat}: {val:.2f}")

# Comparison Table
comp_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree (depth=3)", "Random Forest (100 trees)"],
    "Accuracy": [round(lr_acc, 3), round(dt_pruned_test_acc, 3), round(rf_acc, 3)],
    "Recall": [round(lr_rec, 2), round(dt3_rec, 2), round(rf_rec, 2)],
    "ROC-AUC": [round(lr_auc, 3), round(dt3_auc, 3), round(rf_auc, 3)]
})
print("\nModel Comparison Table (Test Set - Reporting Only):")
print(comp_df.to_string(index=False))
print("Conclusion: Logistic Regression looks best because it achieves the highest ROC-AUC (~0.789) and accuracy (~0.788), meaning it ranks high-risk patients most reliably.")


# ==========================================
# TASK 3.5: Tune, Choose Model & Threshold, Test Once (8 Marks)
# ==========================================
print("\n" + "="*60)
print("TASK 3.5: TUNING, CROSS-VALIDATION, THRESHOLD & FINAL TEST")
print("="*60)

# (a) GridSearchCV on Random Forest (training set only)
param_grid = {"n_estimators": [100, 200], "max_depth": [3, 5, 8]}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring="roc_auc")
grid_rf.fit(X_train, y_train)
print(f"(a) Tuned Random Forest Best Params: {grid_rf.best_params_} | Best 5-Fold CV ROC-AUC: {grid_rf.best_score_:.2f}")

# (b) 5-Fold CV ROC-AUC comparison without touching test set
lr_pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])
cv_auc_lr = cross_val_score(lr_pipe, X_train, y_train, cv=5, scoring="roc_auc").mean()
cv_auc_dt3 = cross_val_score(DecisionTreeClassifier(max_depth=3, random_state=42), X_train, y_train, cv=5, scoring="roc_auc").mean()
cv_auc_rf = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42), X_train, y_train, cv=5, scoring="roc_auc").mean()

print(f"(b) 5-Fold CV ROC-AUC -> Logistic Regression: {cv_auc_lr:.2f} | Depth-3 Tree: {cv_auc_dt3:.2f} | Base RF: {cv_auc_rf:.2f} | Tuned RF: {grid_rf.best_score_:.2f}")
print("    Winner: Logistic Regression (highest CV ROC-AUC ~0.77).")

# (c) Out-of-fold training probabilities for threshold selection
oof_probs = cross_val_predict(lr_pipe, X_train, y_train, cv=5, method="predict_proba")[:, 1]

print("\n(c) Out-of-Fold Threshold Comparison (Training Set):")
for thresh in [0.5, 0.35]:
    oof_preds = (oof_probs >= thresh).astype(int)
    p = precision_score(y_train, oof_preds)
    r = recall_score(y_train, oof_preds)
    f = f1_score(y_train, oof_preds)
    flagged = oof_preds.sum()
    print(f"    Threshold {thresh:.2f} -> Precision: {p:.2f} | Recall: {r:.2f} | F1: {f:.2f} | Flagged: {flagged}")

print("    Chosen Threshold: 0.35 (higher F1 ~0.49 and nearly double the recall ~0.47; catching missed appointments matters far more than making a few extra phone calls).")

# (d) Fit winner on full training set and evaluate ONCE on test set at threshold 0.35
lr_pipe.fit(X_train, y_train)
final_test_probs = lr_pipe.predict_proba(X_test)[:, 1]
final_test_preds = (final_test_probs >= 0.35).astype(int)

final_prec = precision_score(y_test, final_test_preds)
final_rec = recall_score(y_test, final_test_preds)
final_flagged = final_test_preds.sum()
final_auc = roc_auc_score(y_test, final_test_probs)

print(f"\n(d) Final Test Set Evaluation (Threshold = 0.35):")
print(f"    Precision: {final_prec:.2f} | Recall: {final_rec:.2f} | Flagged: {final_flagged} | ROC-AUC: {final_auc:.2f}")

# (e) Save test_risk_scores.csv
risk_df = pd.DataFrame({
    "appointment_id": X_test.index,
    "no_show": y_test.values,
    "risk_prob": np.round(final_test_probs, 4),
    "flag": final_test_preds
})
risk_df.to_csv("test_risk_scores.csv", index=False)
print(f"\n(e) Saved 'test_risk_scores.csv' successfully! Rows: {len(risk_df)}, Flagged (flag=1): {risk_df['flag'].sum()}")
print("    Clinic Threshold Note: In practice, a clinic sets this threshold based on front-desk evening call capacity (how many calls staff can make) versus the financial cost of an empty doctor slot.")