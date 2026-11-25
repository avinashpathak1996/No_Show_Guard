import csv

# यह कोड data फोल्डर के अंदर रखी appointment.csv फाइल को पढ़ेगा
filename = "Data/appointment.csv"

with open(filename, "r", encoding="utf-8") as f:
  reader = csv.reader(f)
  header = next(reader)  # यह फाइल की सबसे ऊपर वाली लाइन (हेडर्स) पढ़ेगा
  print(" फाइल का नाम (Header):", header)