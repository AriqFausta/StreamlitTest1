"""Logika data & model (tanpa dependensi Streamlit) untuk prediksi konsumsi energi."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

DATA_URL = (
    "https://raw.githubusercontent.com/gantarimentari/"
    "energyDataset/main/energydata_complete.csv"
)
TARGET = "Appliances"
DROP_COLS = [TARGET, "date"]

FEATURE_INFO = {
    "lights": "Konsumsi energi lampu (Wh)",
    "T_out": "Suhu luar, stasiun cuaca (°C)",
    "Press_mm_hg": "Tekanan udara (mm Hg)",
    "RH_out": "Kelembapan luar, stasiun cuaca (%)",
    "Windspeed": "Kecepatan angin (m/s)",
    "Visibility": "Jarak pandang (km)",
    "Tdewpoint": "Suhu titik embun (°C)",
    "rv1": "Variabel acak 1",
    "rv2": "Variabel acak 2",
}
ROOMS = {
    1: "Dapur", 2: "Ruang keluarga", 3: "Ruang mencuci", 4: "Ruang kerja",
    5: "Kamar mandi", 6: "Luar bangunan (utara)", 7: "Ruang menyetrika",
    8: "Kamar remaja kedua", 9: "Kamar orang tua",
}
for i, room in ROOMS.items():
    FEATURE_INFO[f"T{i}"] = f"Suhu {room} (°C)"
    FEATURE_INFO[f"RH_{i}"] = f"Kelembapan {room} (%)"


def get_models():
    """Model dengan hyperparameter yang sama seperti di notebook."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        ),
    }


def prepare(df: pd.DataFrame):
    """Pisahkan fitur (semua kolom numerik selain Appliances & date) dan target."""
    missing = [c for c in [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).astype(float)
    y = df[TARGET].astype(float)
    return X, y


def train_and_evaluate(df: pd.DataFrame):
    """Latih semua model (split 70/30, random_state=42) dan hitung metrik."""
    X, y = prepare(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    trained, rows, preds = {}, [], {}
    for name, reg in get_models().items():
        reg.fit(X_train.values, y_train.values)
        y_pred = reg.predict(X_test.values)
        mse = mean_squared_error(y_test, y_pred)
        rows.append({
            "Model": name,
            "R2 Training": reg.score(X_train.values, y_train.values),
            "R2 Testing": r2_score(y_test, y_pred),
            "MSE": mse,
            "RMSE (Wh)": float(np.sqrt(mse)),
            "R": float(np.corrcoef(y_test, y_pred)[0, 1]),
        })
        trained[name] = reg
        preds[name] = y_pred[:50]
    results = (
        pd.DataFrame(rows).sort_values("RMSE (Wh)").reset_index(drop=True)
    )
    return {
        "models": trained,
        "results": results,
        "features": list(X.columns),
        "actual_sample": y_test.values[:50],
        "pred_sample": preds,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
