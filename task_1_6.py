import numpy as np
import pandas as pd

# 1. Load and clean appointments dataset
df = pd.read_csv("Data/appointment.csv")
df.columns = df.columns.str.strip()

# Pass 1 cleaning
df = df.drop_duplicates()
df["booking_channel"] = df["booking_channel"].str.strip().str.title()

date_col = "appt_date" if "appt_date" in df.columns else "app_date"
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df["booked_on"] = pd.to_datetime(df["booked_on"], errors="coerce")
df = df.dropna(subset=[date_col, "status"]).copy()

# Pass 2 cleaning (Validity, gaps, outliers)
df.loc[(df["age"] < 1) | (df["age"] > 100), "age"] = np.nan
df["age"] = df["age"].fillna(df["age"].median()).astype(int)
df["distance_km"] = df["distance_km"].fillna(df["distance_km"].median())

for col in ["distance_km", "fee_inr"]:
  q1 = df[col].quantile(0.25)
  q3 = df[col].quantile(0.75)
  iqr = q3 - q1
  lower_fence = q1 - 1.5 * iqr
  upper_fence = q3 + 1.5 * iqr
  df[col] = np.clip(df[col], lower_fence, upper_fence)

df["distance_km"] = df["distance_km"].round(1)
df["fee_inr"] = df["fee_inr"].round(0).astype(int)

# Feature creation
df["lead_days"] = (df[date_col] - df["booked_on"]).dt.days
df["no_show"] = np.where(df["status"].str.strip() == "No-Show", 1, 0)

# Sort by appointment_id and format dates as YYYY-MM-DD
df = df.sort_values("appointment_id")
df["booked_on"] = df["booked_on"].dt.strftime("%Y-%m-%d")
df["appt_date"] = df[date_col].dt.strftime("%Y-%m-%d")

# Ensure exact 16 columns in required order
cols_order = [
    "appointment_id",
    "patient_id",
    "clinic_id",
    "booked_on",
    "appt_date",
    "age",
    "gender",
    "distance_km",
    "fee_inr",
    "booking_channel",
    "visit_type",
    "reminder_sent",
    "prior_no_shows",
    "status",
    "lead_days",
    "no_show",
]
df = df[cols_order]

# Verify uniqueness and zero nulls before exporting
assert (
    df["appointment_id"].nunique() == len(df)
), "Duplicate appointment_id found!"
assert df.isnull().sum().sum() == 0, "Null values found!"
print(
    f"Verification passed: {len(df)} unique rows, {df.shape[1]} columns, 0"
    " missing values."
)

# Export clean CSV
df.to_csv("appointments_clean.csv", index=False)

# 2. Load enriched clinics data
clinics_df = pd.read_csv("clinics_enriched.csv")
clinics_df.columns = clinics_df.columns.str.strip()

id_col = "clinic_id" if "clinic_id" in clinics_df.columns else "id"
name_col = (
    "clinic_name"
    if "clinic_name" in clinics_df.columns
    else ("name" if "name" in clinics_df.columns else clinics_df.columns[1])
)

# 3. Write load_data.sql script
with open("load_data.sql", "w", encoding="utf-8") as f:
  # Write clinics INSERT statements first
  for _, row in clinics_df.iterrows():
    temp_val = "NULL" if pd.isna(row["temp_c"]) else row["temp_c"]
    c_name = str(row[name_col]).replace("'", "''")
    f.write(
        f"INSERT INTO clinics VALUES ('{row[id_col]}', '{c_name}',"
        f" '{row['city']}', '{row['tier']}', {row['latitude']},"
        f" {row['longitude']}, {temp_val});\n"
    )

  # Write appointments INSERT statements
  for _, row in df.iterrows():
    f.write(
        "INSERT INTO appointments VALUES ("
        f"'{row['appointment_id']}', '{row['patient_id']}',"
        f" '{row['clinic_id']}', '{row['booked_on']}',"
        f" '{row['appt_date']}', {row['age']}, '{row['gender']}',"
        f" {row['distance_km']}, {row['fee_inr']},"
        f" '{row['booking_channel']}', '{row['visit_type']}',"
        f" '{row['reminder_sent']}', {row['prior_no_shows']},"
        f" '{row['status']}', {row['lead_days']}, {row['no_show']});\n"
    )

print("Success! 'appointments_clean.csv' and 'load_data.sql' created.")