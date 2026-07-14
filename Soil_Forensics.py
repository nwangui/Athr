import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import joblib
import json


# ==========================================================
# 1. VINCENTY GEODESIC DISTANCE FUNCTION (WGS-84 VECTORIZED)
# ==========================================================
def vincenty_distance(lat1, lon1, lat2, lon2):
    """
    Computes geodesic distance between coordinate matrices using Vincenty's Formulae
    mapped precisely to the WGS-84 oblate spheroid framework. Fully vectorized for arrays.
    """
    a_axis = 6378137.0
    f_flat = 1 / 298.257223563
    b_axis = (1 - f_flat) * a_axis

    phi1, lam1 = np.radians(lat1), np.radians(lon1)
    phi2, lam2 = np.radians(lat2), np.radians(lon2)

    u1 = np.arctan((1 - f_flat) * np.tan(phi1))
    u2 = np.arctan((1 - f_flat) * np.tan(phi2))
    L = lam2 - lam1

    sin_u1, cos_u1 = np.sin(u1), np.cos(u1)
    sin_u2, cos_u2 = np.sin(u2), np.cos(u2)

    lam_val = np.copy(L)

    for _ in range(100):
        sin_lam, cos_lam = np.sin(lam_val), np.cos(lam_val)
        sin_sigma = np.sqrt((cos_u2 * sin_lam) ** 2 + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lam) ** 2)

        cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lam
        sigma = np.arctan2(sin_sigma, cos_sigma)

        sin_alpha = np.where(sin_sigma == 0, 0, cos_u1 * cos_u2 * sin_lam / sin_sigma)
        cos2_alpha = 1 - sin_alpha ** 2

        cos2_sigma_m = np.where(cos2_alpha == 0, 0, cos_sigma - 2 * sin_u1 * sin_u2 / cos2_alpha)

        C = (f_flat / 16) * cos2_alpha * (4 + f_flat * (4 - 3 * cos2_alpha))
        lam_prev = np.copy(lam_val)
        lam_val = L + (1 - C) * f_flat * sin_alpha * (
                sigma + C * sin_sigma * (cos2_sigma_m + C * cos_sigma * (-1 + 2 * cos2_sigma_m ** 2)))

        if np.all(np.abs(lam_val - lam_prev) < 1e-12):
            break

    u2_scale = cos2_alpha * (a_axis ** 2 - b_axis ** 2) / (b_axis ** 2)
    A = 1 + (u2_scale / 16384) * (4096 + u2_scale * (-768 + u2_scale * (320 - 175 * u2_scale)))
    B = (u2_scale / 1024) * (256 + u2_scale * (-128 + u2_scale * (74 - 47 * u2_scale)))

    delta_sigma = B * sin_sigma * (cos2_sigma_m + (B / 4) * (
            cos_sigma * (-1 + 2 * cos2_sigma_m ** 2) - (B / 6) * cos2_sigma_m * (-3 + 4 * sin_sigma ** 2) * (
            -3 + 4 * cos2_sigma_m ** 2)))

    distance_km = (b_axis * A * (sigma - delta_sigma)) / 1000.0
    return np.where(sin_sigma == 0, 0.0, distance_km)


# ==========================================
# 2. LIVE EXCEL DATABASE LOADING & ROUTING
# ==========================================
try:
    df = pd.read_excel("UAE Soil Database.xlsx")
    print(" Success: Loaded 'UAE Soil Database.xlsx' successfully.")
except FileNotFoundError:
    print("❌ Error: 'UAE Soil Database.xlsx' not found.")
    exit()

df[['Latitude', 'Longitude']] = df['location coordinates'].str.split(',', expand=True).astype(float)

feature_columns = [
    'Si (wt.%)', 'Mg (wt.%)', 'Al (wt.%)', 'Fe (wt.%)', 'Ca (wt.%)',
    'Ti (wt.%)', 'Sr (wt.%)', 'S (wt.%)', 'Mn (wt.%)', 'Cr (wt.%)',
    'Rh (wt.%)', 'Sc (wt.%)', 'Zr (wt.%)', 'K (wt.%)', 'P (wt.%)',
    'S1 (wt.%)','Ni (wt.%)', 'pH'
]

feature_columns = [col for col in feature_columns if col in df.columns]

for col in feature_columns:
    df[col] = df[col].fillna(0.0)

if 'pH' in df.columns:
    df['pH'] = df['pH'].replace(0.0, 8.0).fillna(8.0)

if 'Zone Description' in df.columns:
    df['Zone Description'] = df['Zone Description'].fillna("General Regional Cluster")

    label_encoder = LabelEncoder()
    df['Zone_Encoded'] = label_encoder.fit_transform(df['Zone Description'])
    joblib.dump(label_encoder, "soil_label_encoder.pkl")

    df[['Zone Description', 'Latitude', 'Longitude']].to_pickle("soil_reference_lookup.pkl")
    print(f" Reference lookup database compiled.")
else:
    print("❌ Error: 'Zone Description' column missing from Excel sheet.")
    exit()

# --------------------------------------------------------
# PARALLEL PIPELINE ROUTING BLOCK
# --------------------------------------------------------
X_cls = df[feature_columns]
y_classifier = df['Zone_Encoded']

# 🔒 FIX: Reset the index completely after dropping missing values
df_reg_clean = df.dropna(subset=['Latitude', 'Longitude']).reset_index(drop=True)
X_reg = df_reg_clean[feature_columns]
y_regressor = df_reg_clean[['Latitude', 'Longitude']]

print(f"✅ Data recovery complete. Rows compiled for Classifier: {len(df)}")
print(f"📍 Rows containing geographic coordinates for Regressor: {len(df_reg_clean)}")

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_classifier, test_size=0.2, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_regressor, test_size=0.2, random_state=42)

# --------------------------------------------------------
# 🛠️ UNIFIED PREPROCESSING (MINMAX SCALER)
# --------------------------------------------------------
unified_scaler = MinMaxScaler()
unified_scaler.fit(X_cls)

joblib.dump(unified_scaler, "soil_scaler.pkl")

X_train_cls_scaled = unified_scaler.transform(X_train_cls)
X_test_cls_scaled = unified_scaler.transform(X_test_cls)

X_train_reg_scaled = unified_scaler.transform(X_train_reg)
X_test_reg_scaled = unified_scaler.transform(X_test_reg)


# ==========================================
# 3. TRAIN BOTH PIPELINES
# ==========================================
print(" Training Hybrid AI Engine (Classifier + Regressor)...")

classifier_model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
classifier_model.fit(X_train_cls_scaled, y_train_cls)

regressor_model = KNeighborsRegressor(n_neighbors=3, weights='distance')
regressor_model.fit(X_train_reg_scaled, y_train_reg)

joblib.dump(regressor_model, "uae_soil_regressor.engine")
joblib.dump(classifier_model, "uae_soil_classifier.engine")
print(" Success: Exported hybrid analytical files to folder.")

# ==========================================
# 4. DIAGNOSTICS & METRIC GENERATION
# ==========================================
reg_preds = regressor_model.predict(X_test_reg_scaled)
mse = mean_squared_error(y_test_reg, reg_preds)

# 🔒 FIX: Extract clean arrays with guaranteed position alignment
true_lats = y_test_reg['Latitude'].to_numpy()
true_lons = y_test_reg['Longitude'].to_numpy()

distances = vincenty_distance(
    true_lats,
    true_lons,
    reg_preds[:, 0],
    reg_preds[:, 1]
)
avg_error = distances.mean()

metrics_payload = {
    "avg_error_km": round(float(avg_error), 2),
    "mse_scale": round(float(mse), 6)
}

with open("model_accuracy_metrics.json", "w") as f:
    json.dump(metrics_payload, f)

print("\n" + "=" * 50)
print("Model Accuracy Metrics (WGS-84 Vincenty Model)")
print("=" * 50)
print(json.dumps(metrics_payload, indent=4))
print("=" * 50 + "\n")

print("✅ Success: Performance metrics shared to JSON.")