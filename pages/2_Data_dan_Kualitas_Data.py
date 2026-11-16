"""Halaman 2 - Data dan Kualitas Data"""

import streamlit as st

from utils import (
    compute_data_quality_messages,
    get_column_role_df,
    inject_custom_css,
    render_global_sidebar,
    render_page_header,
    rename_for_display,
)

st.set_page_config(page_title="Data dan Kualitas Data", page_icon="📁", layout="wide")
inject_custom_css()
render_global_sidebar()

render_page_header("2. Data dan Kualitas Data", "Menjelaskan dasar data yang digunakan untuk menjawab seluruh rumusan masalah.")

raw_df = st.session_state.get("raw_df")
processed_df = st.session_state.get("processed_df")
df_work = st.session_state.get("df_work")
warnings_data = st.session_state.get("quality_warnings")

if raw_df is None or processed_df is None or warnings_data is None:
    st.error("Data belum dimuat. Pastikan data default tersedia atau unggah CSV dengan kolom wajib yang sesuai (lihat sidebar).")
    st.stop()

# Peran kolom
st.subheader("Peran Tiap Kolom")
st.markdown(
    "Kolom dibedakan menjadi **pembentuk klaster** (variabel numerik yang distandardisasi untuk K-Means) "
    "dan **pelengkap profil** (untuk deskripsi dan komposisi segmen)."
)
st.dataframe(get_column_role_df(), use_container_width=True, hide_index=True)

# Ukuran data
st.subheader("Ukuran Data")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Baris (mentah)", len(raw_df))
c2.metric("Baris (setelah prapemrosesan)", len(processed_df))
c3.metric("Baris (data kerja/filter aktif)", len(df_work) if df_work is not None else 0)
c4.metric("Jumlah Kolom", len(processed_df.columns))

# Pratinjau tabel
st.subheader("Pratinjau Tabel")
preview_opt = st.radio(
    "Pilih tampilan data",
    options=["Data kerja (setelah filter)", "Data setelah prapemrosesan", "Data mentah"],
    horizontal=True,
)
if preview_opt == "Data mentah":
    prev_df = raw_df
elif preview_opt == "Data setelah prapemrosesan":
    prev_df = processed_df
else:
    prev_df = df_work if df_work is not None else processed_df
n_rows = st.slider("Jumlah baris pratinjau", 5, 100, 20, 5)
st.dataframe(rename_for_display(prev_df.head(n_rows)), use_container_width=True)

# Statistik deskriptif
st.subheader("Statistik Deskriptif")
num_cols = processed_df.select_dtypes(include=["number"]).columns.tolist()
desc = processed_df[num_cols].describe().T
desc = desc.rename(index={
    "CustomerID": "ID Pelanggan",
    "Age": "Usia",
    "Annual_Income($)": "Pendapatan Tahunan ($)",
    "Spending Score(1-100)": "Skor Belanja (1-100)",
    "Work_Experience": "Pengalaman Kerja (tahun)",
    "Family_Size": "Ukuran Keluarga",
})
st.dataframe(desc, use_container_width=True)

# Cek missing & duplikat
st.subheader("Cek Missing Value dan Duplikat")
col_a, col_b = st.columns(2)
with col_a:
    st.write("**Missing value (data mentah):**")
    miss_raw = raw_df.isna().sum().rename("Jumlah Missing").to_frame()
    st.dataframe(miss_raw, use_container_width=True)
with col_b:
    st.write("**Missing value (setelah prapemrosesan, Profession → Unknown):**")
    miss_proc = processed_df.isna().sum().rename("Jumlah Missing").to_frame()
    st.dataframe(miss_proc, use_container_width=True)
    st.write(f"**Jumlah baris duplikat (mentah):** {int(raw_df.duplicated().sum())}")

# Ringkasan peringatan kualitas data
st.subheader("Ringkasan Peringatan Kualitas Data")
for msg in compute_data_quality_messages(warnings_data):
    st.warning(msg)

if st.session_state.get("use_age_filter") and st.session_state.get("age_range"):
    ar = st.session_state["age_range"]
    st.info(
        f"Filter usia aktif: {ar[0]}–{ar[1]}. "
        f"Data kerja yang dianalisis: {len(df_work) if df_work is not None else 0} dari {len(processed_df)} baris."
    )

st.markdown(
    """
<div class="section-card">
<b>Bagaimana data menjadi dasar menjawab rumusan masalah:</b><br>
- Variabel numerik (Usia, Pendapatan Tahunan, Skor Belanja, Pengalaman Kerja, Ukuran Keluarga) menjadi dasar pembentukan klaster (RM1/Tujuan 1).<br>
- Kualitas data dan karakteristik sebarannya memengaruhi evaluasi jumlah klaster & interpretasi Silhouette (RM2/Tujuan 2).<br>
- Atribut numerik beserta Gender dan Profesi digunakan untuk memprofilkan segmen (RM3/Tujuan 3) dan merumuskan strategi pemasaran (RM4/Tujuan 4).<br>
- Seluruh temuan pada halaman berikut dihitung langsung dari data kerja yang telah dipraproses secara transparan.
</div>
""",
    unsafe_allow_html=True,
)
