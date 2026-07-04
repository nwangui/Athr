import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import json


# ==========================================
# 1. HAVERSINE DISTANCE FUNCTION
# ==========================================
def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * 6371.0


# ==========================================
# 2. LIVE EXCEL DATABASE LOADING
# ==========================================
try:
    df = pd.read_excel("UAE Soil Database.xlsx")
    print(" Success: Loaded 'UAE Soil Database.xlsx' successfully.")
except FileNotFoundError:
    print("❌ Error: 'UAE Soil Database.xlsx' not found.")
    exit()

# Parse Coordinates for the Regressor
df[['Latitude', 'Longitude']] = df['location coordinates'].str.split(',', expand=True).astype(float)

feature_columns = [
    'Si (wt.%)', 'Mg (wt.%)', 'Al (wt.%)', 'Fe (wt.%)', 'Ca (wt.%)',
    'Ti (wt.%)', 'Sr (wt.%)', 'S (wt.%)', 'Mn (wt.%)', 'Cr (wt.%)',
    'Rh (wt.%)', 'Sc (wt.%)', 'Zr (wt.%)', 'K (wt.%)', 'P (wt.%)',
    'S1 (wt.%)', 'pH'
]

feature_columns = [col for col in feature_columns if col in df.columns]

for col in feature_columns:
    df[col] = df[col].fillna(0.0)

if 'pH' in df.columns:
    df['pH'] = df['pH'].replace(0.0, 8.0).fillna(8.0)

# Build Target Outputs
if 'Zone Description' in df.columns:
    df['Zone Description'] = df['Zone Description'].fillna("General Regional Cluster")

    # Fit Label Encoder for Classifier
    label_encoder = LabelEncoder()
    df['Zone_Encoded'] = label_encoder.fit_transform(df['Zone Description'])
    joblib.dump(label_encoder, "soil_label_encoder.pkl")

    # Compile reference lookup file
    df[['Zone Description', 'Latitude', 'Longitude']].to_pickle("soil_reference_lookup.pkl")
    print(f" Reference lookup database compiled.")
else:
    print("❌ Error: 'Zone Description' column missing from Excel sheet.")
    exit()

X = df[feature_columns]
y_regressor = df[['Latitude', 'Longitude']]
y_classifier = df['Zone_Encoded']

print(f"✅ Data recovery complete. Rows found: {len(df)}")

# --- Split Using Shared Random States for Alignment ---
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_regressor, test_size=0.2, random_state=42)
_, _, y_train_cls, y_test_cls = train_test_split(X, y_classifier, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 3. TRAIN BOTH PIPELINES
# ==========================================
print(" Training Hybrid AI Engine (Classifier + Regressor)...")

# Train Regressor for unknown coordinates mapping
regressor_model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
regressor_model.fit(X_train_scaled, y_train_reg)

# Train Classifier for strict label boundary detection
classifier_model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
classifier_model.fit(X_train_scaled, y_train_cls)

# Save all assets to disk
joblib.dump(regressor_model, "uae_soil_regressor.engine")
joblib.dump(classifier_model, "uae_soil_classifier.engine")
joblib.dump(scaler, "soil_scaler.pkl")
print(" Success: Exported hybrid analytical files to folder.")

# ==========================================
# 4. DIAGNOSTICS & METRIC GENERATION
# ==========================================
reg_preds = regressor_model.predict(X_test_scaled)
mse = mean_squared_error(y_test_reg, reg_preds)
distances = haversine_distance(y_test_reg['Latitude'].values, y_test_reg['Longitude'].values, reg_preds[:, 0],
                               reg_preds[:, 1])
avg_error = distances.mean()

metrics_payload = {
    "avg_error_km": round(float(avg_error), 2),
    "mse_scale": round(float(mse), 6)
}

with open("model_accuracy_metrics.json", "w") as f:
    json.dump(metrics_payload, f)
print("✅ Success: Performance metrics shared to JSON.")