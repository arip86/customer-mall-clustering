"""Halaman 8 - Kesimpulan dan Unduh"""

import io

import numpy as np
import streamlit as st

from utils import (
    ALIAS_DISPLAY,
    DEFAULT_FEATURES,
    DEFAULT_K,
    RANDOM_STATE,
    build_auto_summary,
    build_research_answers,
    ensure_cluster_results,
    generate_marketing_recommendations,
    get_feature_columns,
    inject_custom_css,
    predict_new_customer,
    render_global_sidebar,
    render_page_header,
    rename_for_display,
)

st.set_page_config(page_title="Kesimpulan dan Unduh", page_icon="✅", layout="wide")
inject_custom_css()
render_global_sidebar()

render_page_header(
    "8. Kesimpulan dan Unduh",
    "Ringkasan temuan otomatis, jawaban rumusan masalah, keterbatasan, unduh hasil, dan prediksi pelanggan baru.",
)

df_work = st.session_state.get("df_work")
if df_work is None or df_work.empty:
    st.error("Data kerja tidak tersedia atau kosong. Periksa sumber data dan filter usia di sidebar.")
    st.stop()

ok, err = ensure_cluster_results()
if not ok:
    st.error(err or "Gagal menyiapkan hasil klaster untuk kesimpulan.")
    st.info("Buka Halaman 4 untuk memilih fitur dan k final, lalu kembali ke halaman ini.")
    st.stop()

df_labeled = st.session_state.get("df_labeled")
cluster_summary = st.session_state.get("cluster_summary")
segment_names = st.session_state.get("segment_names", {})
cluster_result = st.session_state.get("cluster_result")
selected_features = st.session_state.get("selected_features", DEFAULT_FEATURES)
k_final = int(st.session_state.get("k_final", DEFAULT_K))
feature_cols = get_feature_columns(selected_features)

if df_labeled is None or cluster_summary is None or cluster_result is None or cluster_summary.empty:
    st.error("Hasil klaster belum lengkap. Silakan periksa Halaman 4–5.")
    st.stop()

attr_cols = [c for c in ["Age", "Annual_Income($)", "Spending Score(1-100)", "Work_Experience", "Family_Size"] if c in df_labeled.columns]
overall_means = df_labeled[attr_cols].mean()
overall_income = float(df_labeled["Annual_Income($)"].mean()) if "Annual_Income($)" in df_labeled.columns else 0.0
overall_spending = float(df_labeled["Spending Score(1-100)"].mean()) if "Spending Score(1-100)" in df_labeled.columns else 0.0
sil = cluster_result.get("silhouette", np.nan)
sil_text = f"{sil:.4f}" if sil == sil else "N/A"

# Ringkasan temuan otomatis
st.subheader("Ringkasan Temuan (Otomatis dari Data)")
summary = build_auto_summary(cluster_summary, k_final, sil, overall_means)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Jumlah Klaster (k)", summary["k"])
c2.metric("Silhouette Score", sil_text)
c3.metric("Data Kerja", f"{len(df_labeled)} baris")
c4.metric("Random State", RANDOM_STATE)

if summary["largest"]:
    st.markdown(
        f"- **Segmen terbesar:** {summary['largest']['nama']} (Klaster {summary['largest']['cluster']}) "
        f"dengan {summary['largest']['jumlah']} pelanggan ({summary['largest']['persentase']}%)."
    )
if summary["smallest"]:
    st.markdown(
        f"- **Segmen terkecil:** {summary['smallest']['nama']} (Klaster {summary['smallest']['cluster']}) "
        f"dengan {summary['smallest']['jumlah']} pelanggan ({summary['smallest']['persentase']}%)."
    )

st.markdown("**Ukuran dan ciri utama tiap segmen:**")
for item in summary["sizes"]:
    st.markdown(
        f"- Klaster {item['cluster']} — **{item['nama']}**: {item['jumlah']} pelanggan ({item['persentase']}%), "
        f"rata-rata Pendapatan ≈ {item['income']:,.0f}, rata-rata Skor Belanja ≈ {item['spending']:.1f}."
    )

# Jawaban rumusan masalah
st.subheader("Jawaban Singkat Rumusan Masalah")
selected_features_display = [ALIAS_DISPLAY.get(a, a) for a in selected_features]
answers = build_research_answers(k_final, selected_features_display, sil, cluster_summary, segment_names)
with st.container(border=True):
    st.markdown(f"**RM1 (Pengelompokan segmen):** {answers['RM1']}")
with st.container(border=True):
    st.markdown(f"**RM2 (k optimal & kualitas):** {answers['RM2']}")
with st.container(border=True):
    st.markdown(f"**RM3 (Profil segmen):** {answers['RM3']}")
with st.container(border=True):
    st.markdown(f"**RM4 (Strategi pemasaran):** {answers['RM4']}")

# Keterbatasan
st.subheader("Keterbatasan")
st.warning(
    f"- **Pemisahan klaster lemah:** Silhouette Score ({sil_text}) cenderung rendah (~0,15–0,17), "
    "sehingga segmen bersifat pengelompokan praktis, bukan kelompok alami yang terpisah tegas.\n"
    "- **Kualitas data usia:** Age berkisar 0–99 dengan sebaran sangat lebar dan terdapat nilai tidak lazim; "
    "nilai-nilai tersebut tidak dihapus otomatis (tersedia filter usia opsional).\n"
    "- **Korelasi variabel lemah:** hubungan linear antar variabel numerik mendekati nol, "
    "sehingga interpretasi segmen perlu hati-hati dan berfokus pada pola multivariat."
)

# Unduh hasil
st.subheader("Unduh Data Berlabel Klaster")
st.caption(f"Menampilkan pratinjau 30 baris pertama dari {len(df_labeled)} baris data berlabel.")
st.dataframe(rename_for_display(df_labeled.head(30)), use_container_width=True)
csv_buffer = io.StringIO()
df_labeled.to_csv(csv_buffer, index=False)
csv_data = csv_buffer.getvalue().encode("utf-8-sig")
st.download_button(
    label="Unduh CSV Data + Label Klaster",
    data=csv_data,
    file_name=f"hasil_segmentasi_k{k_final}.csv",
    mime="text/csv",
)

# Prediksi pelanggan baru
st.subheader("Prediksi Segmen Pelanggan Baru")
st.caption("Prediksi menggunakan model, scaler, dan fitur aktif yang sama dengan analisis saat ini.")

def _median(col: str, fallback: float) -> float:
    if col in df_work.columns:
        try:
            return float(df_work[col].median())
        except Exception:
            return fallback
    return fallback

with st.form("predict_form"):
    c1, c2 = st.columns(2)
    with c1:
        age_in = st.number_input("Usia", min_value=0, max_value=120, value=int(_median("Age", 30)), step=1)
        income_in = st.number_input("Pendapatan Tahunan ($)", min_value=0, max_value=10_000_000, value=int(_median("Annual_Income($)", 50000)), step=1000)
        spending_in = st.number_input("Skor Belanja (1-100)", min_value=0, max_value=100, value=int(_median("Spending Score(1-100)", 50)), step=1)
    with c2:
        work_in = st.number_input("Pengalaman Kerja (tahun)", min_value=0, max_value=60, value=int(_median("Work_Experience", 5)), step=1)
        family_in = st.number_input("Ukuran Keluarga", min_value=1, max_value=20, value=int(_median("Family_Size", 3)), step=1)
    do_pred = st.form_submit_button("Prediksi Segmen")

if do_pred:
    try:
        full_input = {
            "Age": float(age_in),
            "Annual_Income($)": float(income_in),
            "Spending Score(1-100)": float(spending_in),
            "Work_Experience": float(work_in),
            "Family_Size": float(family_in),
        }
        pred_cluster = predict_new_customer(cluster_result["model"], cluster_result["scaler"], full_input, feature_cols)
        pred_name = segment_names.get(pred_cluster, f"Klaster {pred_cluster}")
        st.success(f"Hasil prediksi: **Klaster {pred_cluster} — {pred_name}**")

        seg_row = cluster_summary[cluster_summary["Cluster"] == pred_cluster]
        if not seg_row.empty:
            row = seg_row.iloc[0]
            st.markdown("**Profil segmen terprediksi:**")
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Rata-rata Usia", f"{float(row.get('Age', 0)):.1f}")
            p2.metric("Rata-rata Pendapatan", f"{float(row.get('Annual_Income($)', 0)):,.0f}")
            p3.metric("Rata-rata Skor Belanja", f"{float(row.get('Spending Score(1-100)', 0)):.1f}")
            p4.metric("Rata-rata Ukuran Keluarga", f"{float(row.get('Family_Size', 0)):.1f}")
            st.write(f"Ukuran segmen: {int(row.get('Jumlah', 0))} pelanggan ({row.get('Persentase', 0)}%)")
            recs = generate_marketing_recommendations(
                float(row.get("Annual_Income($)", 0)),
                float(row.get("Spending Score(1-100)", 0)),
                float(row.get("Family_Size", 0)),
                overall_income,
                overall_spending,
                str(row.get("Profesi_Dominan", "-")),
            )
            st.markdown("**Rekomendasi pemasaran:**")
            for r in recs:
                st.markdown(f"- {r}")
    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {e}")
