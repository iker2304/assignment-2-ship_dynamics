import pandas as pd

df = pd.read_csv("P1_completo.csv", parse_dates=["time"])

time_ = df["time"]
latitude = df["latitude"]
longitude = df["longitude"]
vhm0 = df["VHM0"]
vtpk = df["VTPK"]
vmdr = df["VMDR"]