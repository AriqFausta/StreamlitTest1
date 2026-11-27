import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from model_utils import DATA_URL, FEATURE_INFO, TARGET, train_and_evaluate

st.set_page_config(page_title="Prediksi Energi Peralatan Rumah Tangga", page_icon="⚡", layout="wide")

st.title("⚡ Prediksi Konsumsi Energi Peralatan Rumah Tangga")
st.caption(
    "Regresi pada dataset Appliances Energy Prediction (UCI). "
    "Target: energi peralatan rumah tangga per interval 10 menit (Wh)."
)


@st.cache_data(show_spinner="Memuat dataset...")
def load_data(source):
    return pd.read_csv(source)


@st.cache_resource(show_spinner="Melatih model (hanya sekali, sekitar 30–60 detik)...")
def get_trained(df):
    return train_and_evaluate(df)


# ---------- Sumber data ----------
with st.sidebar:
    st.header("Sumber data")
    uploaded = st.file_uploader("Opsional: unggah energydata_complete.csv", type="csv")
    st.markdown(f"Default: dataset dari [GitHub]({DATA_URL}).")

try:
    df = load_data(uploaded if uploaded is not None else DATA_URL)
except Exception as e:
    st.error(f"Gagal memuat dataset: {e}")
    st.info("Unggah file energydata_complete.csv melalui sidebar.")
    st.stop()

try:
    bundle = get_trained(df)
except Exception as e:
    st.error(f"Gagal melatih model: {e}")
    st.stop()

models, results, features = bundle["models"], bundle["results"], bundle["features"]
best_name = results.loc[0, "Model"]

tab_pred, tab_eval, tab_data = st.tabs(["🔮 Prediksi", "📊 Evaluasi Model", "🗂️ Data"])

# ---------- Prediksi ----------
with tab_pred:
    st.subheader("Masukkan kondisi rumah")
    st.write(
        f"Nilai awal diisi dengan median dataset. Model terbaik (RMSE terendah): **{best_name}**."
    )
    with st.form("prediksi"):
        cols = st.columns(3)
        values = {}
        for i, f in enumerate(features):
            s = df[f].astype(float)
            lo, hi = float(s.min()), float(s.max())
            with cols[i % 3]:
                values[f] = st.number_input(
                    f"{f} — {FEATURE_INFO.get(f, '')}",
                    min_value=lo, max_value=hi,
                    value=float(s.median()), step=0.1, format="%.2f",
                    key=f"in_{f}",
                )
        go = st.form_submit_button("Prediksi", type="primary")

    if go:
        row = pd.DataFrame([values])[features].values
        out = pd.DataFrame(
            {
                "Model": list(models),
                f"Prediksi {TARGET} (Wh)": [
                    max(float(m.predict(row)[0]), 0.0) for m in models.values()
                ],
            }
        )
        best_pred = out.loc[out["Model"] == best_name, out.columns[1]].iloc[0]
        st.metric(f"Prediksi model terbaik ({best_name})", f"{best_pred:.1f} Wh")
        st.dataframe(out.round(2), hide_index=True, use_container_width=True)
        st.caption("Prediksi negatif dipotong ke 0 karena energi tidak bisa negatif.")

# ---------- Evaluasi ----------
with tab_eval:
    st.subheader("Perbandingan model (data testing)")
    st.write(
        f"Split acak 70/30, random_state=42 — {bundle['n_train']:,} data training, "
        f"{bundle['n_test']:,} data testing, {len(features)} fitur."
    )
    st.dataframe(results.round(4), hide_index=True, use_container_width=True)
    st.caption("R² bukan persentase akurasi dan bisa bernilai negatif. RMSE dalam Wh.")

    choice = st.selectbox("Grafik aktual vs prediksi (50 sampel pertama)", list(models))
    fig, ax = plt.subplots(figsize=(12, 3.5))
    pd.DataFrame(
        {"real values": bundle["actual_sample"], "predicted values": bundle["pred_sample"][choice]}
    ).plot(kind="bar", ax=ax)
    ax.set_title(f"{choice}: Real vs Predicted")
    ax.set_xlabel("Sampel ke-i")
    ax.set_ylabel("Energi per 10 menit (Wh)")
    fig.tight_layout()
    st.pyplot(fig)

# ---------- Data ----------
with tab_data:
    st.subheader("Contoh data")
    st.dataframe(df.head(50), use_container_width=True)
    st.subheader("Statistik deskriptif")
    st.dataframe(df.describe().T, use_container_width=True)
