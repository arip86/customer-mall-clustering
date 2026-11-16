"""Halaman 4 - Menentukan Jumlah Klaster (Rumusan Masalah 2 / Tujuan 2)"""

import numpy as np
import plotly.express as px
import streamlit as st

from utils import (
    ALIAS_DISPLAY,
    DEFAULT_FEATURES,
    DEFAULT_K,
    NUMERIC_FEATURE_OPTIONS,
    RANDOM_STATE,
    evaluate_k_range,
    get_feature_columns,
    inject_custom_css,
    prepare_features,
    render_global_sidebar,
    render_page_header,
    run_kmeans,
)

st.set_page_config(page_title="Menentukan Jumlah Klaster", page_icon="🔎", layout="wide")
inject_custom_css()
render_global_sidebar()

render_page_header("4. Menentukan Jumlah Klaster", "Menjawab Rumusan Masalah 2 / Tujuan 2: jumlah klaster optimal dan kualitas pemisahan.")

df_work = st.session_state.get("df_work")
if df_work is None or df_work.empty:
    st.error("Data kerja tidak tersedia atau kosong. Periksa sumber data dan filter usia di sidebar.")
    st.stop()

st.subheader("Pemilihan Fitur dan Standardisasi")
current_features = st.session_state.get("selected_features", DEFAULT_FEATURES)
selected_features = st.multiselect(
    "Pilih fitur untuk klastering (distandardisasi dengan StandardScaler)",
    options=NUMERIC_FEATURE_OPTIONS,
    default=[f for f in current_features if f in NUMERIC_FEATURE_OPTIONS] or DEFAULT_FEATURES,
    help="Pilih minimal 2 fitur numerik.",
)
st.session_state["selected_features"] = selected_features

feature_cols = get_feature_columns(selected_features)
if len(feature_cols) < 2:
    st.error("Pilih minimal 2 fitur untuk melanjutkan klastering.")
    st.stop()

try:
    feat_df = prepare_features(df_work, feature_cols)
except Exception as e:
    st.error(f"Gagal menyiapkan fitur: {e}")
    st.stop()

st.success(f"Fitur aktif: {', '.join([ALIAS_DISPLAY.get(a, a) for a in selected_features])} (distandardisasi dengan StandardScaler).")

# Evaluasi k = 2..10
st.subheader("Evaluasi Jumlah Klaster (k = 2 sampai 10)")
max_k_eval = min(10, len(feat_df) - 1)
if max_k_eval < 2:
    st.error("Jumlah data terlalu sedikit untuk evaluasi k (minimal butuh > 2 baris).")
    st.stop()

try:
    eval_df = evaluate_k_range(feat_df, k_min=2, k_max=max_k_eval, random_state=RANDOM_STATE)
except Exception as e:
    st.error(f"Gagal menghitung metrik evaluasi k: {e}")
    st.stop()

e1, e2 = st.columns(2)
with e1:
    fig_elbow = px.line(
        eval_df, x="k", y="inertia", markers=True,
        title="Grafik Elbow (Inertia)",
        labels={"k": "Jumlah Klaster (k)", "inertia": "Inertia"},
    )
    st.plotly_chart(fig_elbow, use_container_width=True)
with e2:
    fig_sil = px.line(
        eval_df, x="k", y="silhouette", markers=True,
        title="Grafik Silhouette Score",
        labels={"k": "Jumlah Klaster (k)", "silhouette": "Silhouette Score"},
    )
    st.plotly_chart(fig_sil, use_container_width=True)

st.dataframe(
    eval_df.rename(columns={"k": "k", "inertia": "Inertia", "silhouette": "Silhouette Score"}),
    use_container_width=True, hide_index=True,
)

# Slider pemilihan k final
st.subheader("Pemilihan Jumlah Klaster Final")
k_current = int(st.session_state.get("k_final", DEFAULT_K))
k_current = max(2, min(k_current, max_k_eval))
k_final = st.slider(
    "Pilih k final (disimpan untuk halaman berikutnya)",
    min_value=2,
    max_value=max_k_eval,
    value=k_current,
    step=1,
)
st.session_state["k_final"] = k_final
st.caption(f"Random state tetap: {RANDOM_STATE} (hasil konsisten). k final tersimpan: {k_final}")

# Hitung metrik untuk k terpilih
try:
    res_k = run_kmeans(feat_df, k=k_final, random_state=RANDOM_STATE)
    sil_k = res_k.get("silhouette", np.nan)
except Exception as e:
    st.error(f"Gagal menghitung hasil untuk k={k_final}: {e}")
    st.stop()

sil_text = f"{sil_k:.4f}" if sil_k == sil_k else "N/A"
m1, m2 = st.columns(2)
m1.metric(f"Silhouette Score (k={k_final})", sil_text)
m2.metric(f"Inertia (k={k_final})", f"{res_k.get('inertia', 0):,.2f}")

st.markdown("#### Penjelasan Kualitas Klaster (Jujur)")
st.warning(
    f"Silhouette Score untuk k={k_final} adalah **{sil_text}**. "
    "Nilai mendekati 1 menandakan klaster terpisah baik, mendekati 0 menandakan tumpang tindih, "
    "dan negatif menandakan kemungkinan penempatan klaster kurang tepat. "
    "Pada data ini nilai cenderung rendah (sekitar 0,15–0,17), sehingga pemisahan antar klaster relatif lemah. "
    "Artinya, segmen yang dihasilkan merupakan **pengelompokan praktis untuk pemasaran**, "
    "bukan kelompok alami yang terpisah tegas."
)

st.success(
    "Keputusan k final telah disimpan di session dan akan digunakan pada Halaman 5–8 "
    "(Hasil Pembagian, Profil Segmen, Rekomendasi, dan Kesimpulan)."
)
