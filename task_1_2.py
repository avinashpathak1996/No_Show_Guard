import csv
import json

# तुम्हारी फाइल का जो सही नाम और फोल्डर है
filename = "Data/appointment.csv"

header = None
rows = []
rejected_rows = []

with open(filename, "r", encoding="utf-8") as f:
  reader = csv.reader(f)
  header = next(reader)
  print("Header Founded:", header)

  for row in reader:
    try:
      # हर row की वैल्यू को सही datatype में बदलना
      parsed_row = {
          "appointment_id": row[0],
          "patient_id": row[1],
          "clinic_id": row[2],
          "booked_on": row[3],
          "app_date": row[4],
          "age": int(row[5]),
          "gender": row[6],
          "distance_km": float(row[7]),
          "fee_inr": int(row[8]),
          "booking_channel": row[9],
          "visit_type": row[10],
          "reminder_sent": row[11],
          "prior_no_shows": int(row[12]),
          "status": row[13],
      }
      rows.append(parsed_row)
    except ValueError as e:
      # अगर कोई गलती या गलत डेटा मिला, तो उसे rejected_rows में डाल देंगे
      rejected_rows.append({"row": row, "error": str(e)})

print("Total Accepted Rows:", len(rows))
print("Total Rejected Rows:", len(rejected_rows))

# Rejected rows ko JSON file me save karna
with open("rejected_rows.json", "w", encoding="utf-8") as jf:
  json.dump(rejected_rows, jf, indent=2)