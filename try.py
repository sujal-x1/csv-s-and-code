import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score

# Load CSVs
logon_df = pd.read_csv(r"D:\g\r6.2\logon.csv")
device_df = pd.read_csv(r"D:\g\r6.2\device.csv")
file_df = pd.read_csv(r"D:\g\r6.2\file.csv")
decoy_df = pd.read_csv(r"D:\g\r6.2\decoy_file.csv")
ldap_df = pd.read_csv(r"D:\g\r6.2\LDAP\2011-05.csv")

# Convert date columns from string to datetime
logon_df["date"] = pd.to_datetime(logon_df["date"], errors='coerce')
device_df["date"] = pd.to_datetime(device_df["date"], errors='coerce')
file_df["date"] = pd.to_datetime(file_df["date"], errors='coerce')

# Drop rows with null dates after conversion
device_df = device_df.dropna(subset=["date"])
file_df = file_df.dropna(subset=["date"])

# Create decoy lookup set
decoy_set = set(decoy_df["decoy_filename"].str.lower())

# Feature: user daily logon count
logon_df["date_only"] = logon_df["date"].dt.date
logon_features = logon_df.groupby(["user", "date_only"]).agg(
    logon_count=("activity", "count")
).reset_index()

# Feature: device connects count per user per day
device_df["date_only"] = device_df["date"].dt.date
device_features = device_df.groupby(["user", "date_only"]).agg(
    device_connects=("activity", "count")
).reset_index()

# Feature: file activity and decoy access count
file_df["is_decoy_access"] = file_df["filename"].str.lower().isin(decoy_set)
file_df["date_only"] = file_df["date"].dt.date
file_features = file_df.groupby(["user", "date_only"]).agg(
    file_ops=("activity", "count"),
    decoy_access=("is_decoy_access", "sum")
).reset_index()

# Merge features (inner join to ensure common (user, date))
features = logon_features.merge(device_features, how="inner", on=["user", "date_only"])
features = features.merge(file_features, how="inner", on=["user", "date_only"])
features = features.fillna(0)

# Final feature matrix for Isolation Forest
X = features[["logon_count", "device_connects", "file_ops", "decoy_access"]]

# Train Isolation Forest
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(X)

# Predict anomalies (1 = normal, -1 = anomaly)
features['anomaly_raw'] = model.predict(X)
features['anomaly_label'] = features['anomaly_raw'].map({1: 0, -1: 1})  # 1 = anomaly

# Print basic output
print("\n--- Sample Predictions ---")
print(features[["user", "date_only", "logon_count", "device_connects", "file_ops", "decoy_access", "anomaly_label"]].head())

print(f"\nTotal anomalies detected: {features['anomaly_label'].sum()}")

# Save detailed output to CSV
features.to_csv(r"D:\g\r6.2\insider_threat_results.csv", index=False)

# Optional: show first few anomaly rows
print("\n--- First 5 Anomalies Detected ---")
print(features[features["anomaly_label"] == 1][["user", "date_only", "logon_count", "device_connects", "file_ops", "decoy_access"]].head())
