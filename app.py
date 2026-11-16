"""
Halaman 1 - Latar Belakang, Rumusan Masalah, dan Tujuan
Laporan interaktif segmentasi pelanggan mall menggunakan K-Means.
Jalankan dengan: streamlit run app.py
"""

import pandas as pd
import streamlit as st

from utils import (
    DEFAULT_FEATURES,
    DEFAULT_K,
    RANDOM_STATE,
    compute_data_quality_messages,
    inject_custom_css,
    render_global_sidebar,
    render_page_header,
)

st.set_page_config(
    page_title="Segmentasi Pelanggan Mall - K-Means",
    page_icon="📊",
    layout="wide",
)

inject_custom_css()
render_global_sidebar()

render_page_header(
    "Laporan Interaktif Segmentasi Pelanggan Mall (K-Means)",
    "Alur penelitian dari latar belakang hingga rekomendasi pemasaran berbasis data.",
)

st.markdown(
    "<div class='soft-info'>Gunakan navigasi halaman di sidebar untuk melihat jawaban tiap rumusan masalah. "
    "Pilih k final di <b>Halaman 4</b> agar hasil klaster pada Halaman 5–8 konsisten.</div>",
    unsafe_allow_html=True,
)
st.write("")

# -----------------------------
# Konten Halaman 1
# -----------------------------
st.header("1. Latar Belakang, Rumusan Masalah, dan Tujuan")

# Latar belakang
st.subheader("Latar Belakang")
st.markdown(
    """
<div class="section-card" style="border-left: 5px solid #2563EB;">
Pelanggan mall sangat beragam dari sisi usia, pendapatan, profesi, pengalaman kerja, dan ukuran keluarga. 
Strategi pemasaran yang seragam membuat promosi kurang tepat sasaran dan anggaran kurang efisien. 
Segmentasi pelanggan dengan <b>clustering</b> memungkinkan pelanggan dikelompokkan berdasarkan kemiripan karakteristik.
</div>
""",
    unsafe_allow_html=True,
)

col_rm, col_tj = st.columns(2)

with col_rm:
    st.subheader("Rumusan Masalah")
    with st.container(border=True):
        st.markdown(
            """
1. Bagaimana pelanggan mall dapat dikelompokkan ke dalam segmen-segmen berbeda berdasarkan usia, pendapatan tahunan, Spending Score, pengalaman kerja, dan ukuran keluarga menggunakan K-Means?
2. Berapa jumlah klaster yang optimal, dan seberapa baik kualitas pemisahan antar klaster?
3. Apa karakteristik dan profil tiap segmen, termasuk komposisi jenis kelamin dan profesinya?
4. Strategi pemasaran apa yang sesuai untuk tiap segmen?
"""
        )

with col_tj:
    st.subheader("Tujuan")
    with st.container(border=True):
        st.markdown(
            """
1. Membentuk segmen pelanggan dengan K-Means dari variabel numerik yang distandardisasi.
2. Menentukan jumlah klaster optimal dengan Elbow dan Silhouette Score serta menilai kualitas klaster.
3. Memprofilkan tiap segmen dari rata-rata atribut dan proporsi gender serta profesi.
4. Merumuskan strategi pemasaran untuk tiap segmen.
"""
        )

st.subheader("Pemetaan Rumusan Masalah → Tujuan → Halaman Jawaban")
mapping_df = pd.DataFrame(
    [
        {
            "Rumusan Masalah": "RM1: Pengelompokan segmen pelanggan dengan K-Means berdasarkan usia, pendapatan, Spending Score, pengalaman kerja, dan ukuran keluarga",
            "Tujuan": "Tujuan 1: Membentuk segmen pelanggan dengan K-Means dari variabel numerik yang distandardisasi",
            "Halaman Jawaban": "Halaman 5 - Hasil Pembagian Klaster (didukung Halaman 3 sebagai gambaran awal)",
        },
        {
            "Rumusan Masalah": "RM2: Jumlah klaster optimal dan kualitas pemisahan antar klaster",
            "Tujuan": "Tujuan 2: Menentukan jumlah klaster optimal dengan Elbow & Silhouette Score serta menilai kualitas klaster",
            "Halaman Jawaban": "Halaman 4 - Menentukan Jumlah Klaster",
        },
        {
            "Rumusan Masalah": "RM3: Karakteristik dan profil tiap segmen (atribut, gender, profesi)",
            "Tujuan": "Tujuan 3: Memprofilkan tiap segmen dari rata-rata atribut dan proporsi gender serta profesi",
            "Halaman Jawaban": "Halaman 6 - Profil Segmen",
        },
        {
            "Rumusan Masalah": "RM4: Strategi pemasaran yang sesuai untuk tiap segmen",
            "Tujuan": "Tujuan 4: Merumuskan strategi pemasaran untuk tiap segmen",
            "Halaman Jawaban": "Halaman 7 - Rekomendasi Pemasaran",
        },
        {
            "Rumusan Masalah": "Ringkasan jawaban & keterbatasan",
            "Tujuan": "Sintesis temuan",
            "Halaman Jawaban": "Halaman 8 - Kesimpulan dan Unduh",
        },
    ]
)
st.dataframe(mapping_df, use_container_width=True, hide_index=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Jumlah Klaster (k) Aktif", st.session_state.get("k_final", DEFAULT_K))
with c2:
    dfw = st.session_state.get("df_work")
    st.metric("Data Kerja Aktif", f"{len(dfw)} baris" if dfw is not None else "-")
with c3:
    st.metric("Random State", RANDOM_STATE)

if st.session_state.get("df_work") is None:
    st.warning("Data belum siap. Pastikan data default tersedia atau unggah CSV dengan kolom wajib yang sesuai.")
else:
    st.success(
        f"Data siap digunakan: {len(st.session_state['df_work'])} baris data kerja aktif "
        f"(dari {len(st.session_state['processed_df'])} baris setelah prapemrosesan). "
        f"Fitur aktif: {', '.join(st.session_state.get('selected_features', DEFAULT_FEATURES))}."
    )
