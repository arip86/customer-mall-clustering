"""Halaman 3 - Gambaran Pelanggan (persiapan Rumusan Masalah 1)"""

import plotly.express as px
import streamlit as st

from utils import DISPLAY_LABELS, inject_custom_css, render_global_sidebar, render_page_header

st.set_page_config(page_title="Gambaran Pelanggan", page_icon="📈", layout="wide")
inject_custom_css()
render_global_sidebar()

render_page_header("3. Gambaran Pelanggan", "Eksplorasi awal karakteristik pelanggan sebagai persiapan menjawab Rumusan Masalah 1.")

df_work = st.session_state.get("df_work")
if df_work is None or df_work.empty:
    st.error("Data kerja tidak tersedia atau kosong. Periksa sumber data dan filter usia di sidebar.")
    st.stop()

# Histogram
st.subheader("Distribusi Variabel Numerik")
h1, h2, h3 = st.columns(3)
with h1:
    fig = px.histogram(df_work, x="Age", nbins=30, title="Histogram Usia", labels={"Age": DISPLAY_LABELS["Age"]})
    st.plotly_chart(fig, use_container_width=True)
with h2:
    fig = px.histogram(
        df_work, x="Annual_Income($)", nbins=30, title="Histogram Pendapatan Tahunan",
        labels={"Annual_Income($)": DISPLAY_LABELS["Annual_Income($)"]},
    )
    st.plotly_chart(fig, use_container_width=True)
with h3:
    fig = px.histogram(
        df_work, x="Spending Score(1-100)", nbins=30, title="Histogram Skor Belanja",
        labels={"Spending Score(1-100)": DISPLAY_LABELS["Spending Score(1-100)"]},
    )
    st.plotly_chart(fig, use_container_width=True)

h4, h5 = st.columns(2)
with h4:
    fig = px.histogram(
        df_work, x="Work_Experience", nbins=20, title="Histogram Pengalaman Kerja",
        labels={"Work_Experience": DISPLAY_LABELS["Work_Experience"]},
    )
    st.plotly_chart(fig, use_container_width=True)
with h5:
    fig = px.histogram(
        df_work, x="Family_Size", nbins=15, title="Histogram Ukuran Keluarga",
        labels={"Family_Size": DISPLAY_LABELS["Family_Size"]},
    )
    st.plotly_chart(fig, use_container_width=True)

# Bar chart gender dan profesi
st.subheader("Komposisi Gender dan Profesi")
b1, b2 = st.columns(2)
with b1:
    g_counts = df_work["Gender"].value_counts().reset_index()
    g_counts.columns = ["Gender", "Jumlah"]
    fig = px.bar(g_counts, x="Gender", y="Jumlah", text="Jumlah", title="Jumlah Pelanggan per Gender",
                 labels={"Gender": DISPLAY_LABELS["Gender"]}, color="Gender")
    st.plotly_chart(fig, use_container_width=True)
with b2:
    p_counts = df_work["Profession"].value_counts().reset_index()
    p_counts.columns = ["Profession", "Jumlah"]
    fig = px.bar(p_counts, x="Profession", y="Jumlah", text="Jumlah", title="Jumlah Pelanggan per Profesi",
                 labels={"Profession": DISPLAY_LABELS["Profession"]})
    fig.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)

# Heatmap korelasi
st.subheader("Korelasi Variabel Numerik")
corr_cols = [c for c in ["Age", "Annual_Income($)", "Spending Score(1-100)", "Work_Experience", "Family_Size"] if c in df_work.columns]
corr = df_work[corr_cols].corr()
corr_disp = corr.copy()
corr_disp.index = [DISPLAY_LABELS.get(i, i) for i in corr_disp.index]
corr_disp.columns = [DISPLAY_LABELS.get(i, i) for i in corr_disp.columns]
fig_corr = px.imshow(
    corr_disp, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
    title="Heatmap Korelasi Variabel Numerik",
)
st.plotly_chart(fig_corr, use_container_width=True)

st.info(
    "Korelasi antar variabel numerik yang lemah (mendekati nol) menunjukkan tidak ada satu hubungan linear dominan. "
    "Oleh karena itu, **clustering multivariat** (K-Means) digunakan untuk mengelompokkan pelanggan berdasarkan "
    "kemiripan pola gabungan beberapa variabel sekaligus, bukan hanya satu dimensi."
)
