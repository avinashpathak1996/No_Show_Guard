import numpy as np
import pandas as pd

# Load the dataset
df = pd.read_csv("Data/appointment.csv")

# Clean column names from any hidden spaces
df.columns = df.columns.str.strip()

# Find the exact appointment date column name (appt_date or app_date)
date_col = "appt_date" if "appt_date" in df.columns else "app_date"

# Convert date columns to datetime
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df["booked_on"] = pd.to_datetime(df["booked_on"], errors="coerce")

# Create lead_days and no_show columns
df["lead_days"] = (df[date_col] - df["booked_on"]).dt.days
df["no_show"] = np.where(df["status"].str.strip() == "No-Show", 1, 0)

# Remove rows where lead_days could not be calculated
df_clean = df.dropna(subset=["lead_days"]).copy()

# Extract specified columns into a NumPy array
x = df_clean[["age", "distance_km", "fee_inr", "lead_days"]].to_numpy()

# (a) Compute the mean of each column
col_means = np.nanmean(x, axis=0)
print("Column Means (Age, Distance, Fee, Lead Days):", col_means)

# (b) Compute number and share of rows with lead_days > 14 using a boolean mask
mask = x[:, 3] > 14
count_gt_14 = mask.sum()
share_gt_14 = mask.mean()
print(f"Lead days > 14 -> Count: {count_gt_14}, Share: {share_gt_14:.4f}")

# (c) Compute z-scores for the distance_km column
dist_col = x[:, 1]
dist_mean = np.nanmean(dist_col)
dist_std = np.nanstd(dist_col)
z_scores = (dist_col - dist_mean) / dist_std
count_z_gt_2 = np.sum(np.abs(z_scores) > 2)
print("Rows with |z-score| > 2 in distance_km:", count_z_gt_2)