import pandas as pd
import requests

# Load the clinics dataset
clinics_df = pd.read_csv("Data/clinic.csv")
clinics_df.columns = clinics_df.columns.str.strip()

# Auto-detect the clinic name column (e.g., 'clinic_name' or 'name')
name_col = (
    "clinic_name"
    if "clinic_name" in clinics_df.columns
    else ("name" if "name" in clinics_df.columns else clinics_df.columns[1])
)

temp_c_list = []

for idx, row in clinics_df.iterrows():
  lat = row["latitude"]
  lon = row["longitude"]
  clinic_label = row[name_col]

  url = "https://api.open-meteo.com/v1/forecast"
  params = {
      "latitude": lat,
      "longitude": lon,
      "hourly": "temperature_2m",
      "forecast_days": 1,
  }

  temp_val = None
  # Retry up to 2 times if the API call fails
  for attempt in range(2):
    try:
      response = requests.get(url, params=params, timeout=10)
      if response.status_code == 200:
        data = response.json()
        if "hourly" in data and "temperature_2m" in data["hourly"]:
          temp_val = data["hourly"]["temperature_2m"][0]
          break
      else:
        if attempt == 1:
          print(
              f"Failed for {clinic_label}: HTTP status {response.status_code}"
          )
    except requests.exceptions.RequestException as e:
      if attempt == 1:
        print(f"Failed for {clinic_label}: {e}")

  temp_c_list.append(temp_val)

# Add the new column and save to CSV
clinics_df["temp_c"] = temp_c_list
clinics_df.to_csv("clinics_enriched.csv", index=False)

print("Task 1.5 completed! 'clinics_enriched.csv' created:")
print(clinics_df[[name_col, "city", "temp_c"]])