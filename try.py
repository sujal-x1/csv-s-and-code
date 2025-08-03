import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score

# Load the necessary CSV files
logon_df = pd.read_csv(r"D:\g\r6.2\logon.csv")
device_df = pd.read_csv(r"D:\g\r6.2\device.csv")
file_df = pd.read_csv(r"D:\g\r6.2\file.csv")
decoy_df = pd.read_csv(r"D:\g\r6.2\decoy_file.csv")
ldap_df = pd.read_csv(r"D:\g\r6.2\LDAP\2011-05.csv")

# Convert string dates to datetime format (some rows might have bad values, so we handle that)
logon_df["date"] = pd.to_datetime(logon_df["date"], errors='coerce')
device_df["date"] = pd.to_datetime(device_df["date"], errors='coerce')
file_df["date"] = pd.to_datetime(file_df["date"], errors='coerce')

# Drop any rows where date couldn’t be parsed
device_df = device_df.dropna(subset=["date"])
file_df = file_df.dropna(subset=["date"])

# Get a set of all known decoy filenames (used later to flag suspicious activity)
decoy_set = set(decoy_df["decoy_filename"].str.lower())

# Count how many times each user logs in per day
logon_df["date_only"] = logon_df["date"].dt.date
logon_features = logon_df.groupby(["user", "date_only"]).agg(
    logon_count=("activity", "count")
).reset_index()

# Count how many devices each user connects to per day
device_df["date_only"] = device_df["date"].dt.date
device_features = device_df.groupby(["user", "date_only"]).agg(
    device_connects=("activity", "count")
).reset_index()

# Count how many files are accessed per day + how many of those were decoy files
file_df["is_decoy_access"] = file_df["filename"].str.lower().isin(decoy_set)
file_df["date_only"] = file_df["date"].dt.date
file_features = file_df.groupby(["user", "date_only"]).agg(
    file_ops=("activity", "count"),
    decoy_access=("is_decoy_access", "sum")
).reset_index()

# Merge all feature data together (only keep rows that exist in all 3 datasets)
features = logon_features.merge(device_features, how="inner", on=["user", "date_only"])
features = features.merge(file_features, how="inner", on=["user", "date_only"])
features = features.fillna(0)

# Build the actual input for the anomaly detection model
X = features[["logon_count", "device_connects", "file_ops", "decoy_access"]]

# Train the Isolation Forest model on our user activity data
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(X)

# Predict which rows are anomalous (1 = normal, -1 = anomaly)
features['anomaly_raw'] = model.predict(X)
features['anomaly_label'] = features['anomaly_raw'].map({1: 0, -1: 1})  # Mark 1 as anomaly

# Preview a few predictions
print("\n--- Sample Predictions ---")
print(features[["user", "date_only", "logon_count", "device_connects", "file_ops", "decoy_access", "anomaly_label"]].head())

print(f"\nTotal anomalies detected: {features['anomaly_label'].sum()}")

# Save all results to a CSV file for further analysis
features.to_csv(r"D:\g\r6.2\insider_threat_results.csv", index=False)

# Print the first few rows where an anomaly was found
print("\n--- First 5 Anomalies Detected ---")
print(features[features["anomaly_label"] == 1][["user", "date_only", "logon_count", "device_connects", "file_ops", "decoy_access"]].head())
