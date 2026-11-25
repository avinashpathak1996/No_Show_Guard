import numpy as np
import pandas as pd

# Load dataset
filename = "Data/appointment.csv"
df = pd.read_csv(filename)

# (a) Treat age below 1 or above 100 as invalid and set to missing (NaN)
df.loc[(df["age"] < 1) | (df["age"] > 100), "age"] = np.nan

# (b) Fill missing age with median and missing distance with median
median_age = df["age"].median()
df["age"] = df["age"].fillna(median_age).astype(int)

median_dist = df["distance_km"].median()
df["distance_km"] = df["distance_km"].fillna(median_dist)

# (c) Compute Q1, Q3, and IQR for distance_km and fee_inr to cap outliers
for col in ["distance_km", "fee_inr"]:
  q1 = df[col].quantile(0.25)
  q3 = df[col].quantile(0.75)
  iqr = q3 - q1
  lower_fence = q1 - 1.5 * iqr
  upper_fence = q3 + 1.5 * iqr

  # Cap the values within fences
  df[col] = np.where(df[col] < lower_fence, lower_fence, df[col])
  df[col] = np.where(df[col] > upper_fence, upper_fence, df[col])

# (d) Round distance_km to 1 decimal and fee_inr to whole rupees
df["distance_km"] = df["distance_km"].round(1)
df["fee_inr"] = df["fee_inr"].round(0).astype(int)

print("Task 1.3 completed successfully!")