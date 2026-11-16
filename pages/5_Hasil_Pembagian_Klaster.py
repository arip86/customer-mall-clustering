"""Halaman 5 - Hasil Pembagian Klaster (Rumusan Masalah 1 / Tujuan 1)"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (
    ALIAS_DISPLAY,
    DEFAULT_FEATURES,
    DEFAULT_K,
    DISPLAY_LABELS,
    RANDOM_STATE,
    ensure_cluster_results,
    get_feature_columns,
    inject_custom_css,
    render_global_sidebar,
    render_page_header,
    rename_for_display,
)

st.set_page_config(page_title="Hasil Pembagian Klaster", page_icon="🧩", layout="wide")
inject_custom_css()
render_global_sidebar()

render_page_header("5. Hasil Pembagian Klaster", "Menjawab Rumusan Masalah 1 / Tujuan 1: visualisasi dan hasil pembagian segmen pelanggan.")

df_work = st.session_state.get("df_work")
if df_work is None or df_work.empty:
    st.error("Data kerja tidak tersedia atau kosong. Periksa sumber data dan filter usia di sidebar.")
    st.stop()

ok, err = ensure_cluster_results()
if not ok:
    st.error(err or "Gagal menyiapkan hasil klaster.")
    st.info("Buka Halaman 4 untuk memilih fitur dan k final, lalu kembali ke halaman ini.")
    st.stop()

cluster_result = st.session_state.get("cluster_result")
cluster_summary = st.session_state.get("cluster_summary")
segment_names = st.session_state.get("segment_names", {})
df_labeled = st.session_state.get("df_labeled")
seg_color_map = st.session_state.get("seg_color_map", {})
selected_features = st.session_state.get("selected_features", DEFAULT_FEATURES)
k_final = int(st.session_state.get("k_final", DEFAULT_K))

st.markdown(
    f"**Konfigurasi aktif:** k = {k_final}, random_state = {RANDOM_STATE}, "
    f"fitur = {', '.join([ALIAS_DISPLAY.get(a,a) for a in selected_features])}"
)
sil = cluster_result.get("silhouette", np.nan)
st.write(f"Silhouette Score (k terpilih): **{sil:.4f}**" if sil == sil else "Silhouette Score: **N/A**")

# PCA 2D
st.subheader("Visualisasi PCA 2D per Klaster")
pca_coords = cluster_result["pca_coords"]
pca_df = pd.DataFrame(pca_coords, columns=["PC1", "PC2"], index=df_work.index)
pca_df["Cluster"] = df_labeled["Cluster"].values
pca_df["Nama Segmen"] = df_labeled["Nama_Segmen"].values
exp = cluster_result.get("pca_explained", [0, 0])
fig_pca = px.scatter(
    pca_df, x="PC1", y="PC2", color="Nama Segmen",
    color_discrete_map=seg_color_map,
    title="PCA 2D Hasil Klastering (diwarnai per segmen)",
    labels={
        "PC1": f"PC1 ({exp[0]*100:.1f}% varians)" if len(exp) > 0 else "PC1",
        "PC2": f"PC2 ({exp[1]*100:.1f}% varians)" if len(exp) > 1 else "PC2",
        "Nama Segmen": "Segmen",
    },
    hover_data=["Cluster"],
)
st.plotly_chart(fig_pca, use_container_width=True)

# Scatter Income vs Spending
st.subheader("Scatter Pendapatan Tahunan vs Skor Belanja per Klaster")
fig_sc = px.scatter(
    df_labeled,
    x="Annual_Income($)", y="Spending Score(1-100)", color="Nama_Segmen",
    color_discrete_map=seg_color_map,
    title="Annual Income vs Spending Score per Segmen",
    labels={
        "Annual_Income($)": DISPLAY_LABELS["Annual_Income($)"],
        "Spending Score(1-100)": DISPLAY_LABELS["Spending Score(1-100)"],
        "Nama_Segmen": "Segmen",
    },
    hover_data=["Cluster", "Age", "Gender", "Profession"],
)
st.plotly_chart(fig_sc, use_container_width=True)

# Bar chart jumlah & persentase
st.subheader("Jumlah dan Persentase Pelanggan per Klaster")
count_df = cluster_summary[["Cluster", "Nama_Segmen", "Jumlah", "Persentase"]].sort_values("Cluster").copy()
c1, c2 = st.columns(2)
with c1:
    fig_count = px.bar(
        count_df, x="Nama_Segmen", y="Jumlah", color="Nama_Segmen",
        color_discrete_map=seg_color_map, text="Jumlah",
        title="Jumlah Pelanggan per Segmen",
        labels={"Nama_Segmen": "Segmen", "Jumlah": "Jumlah"},
    )
    fig_count.update_layout(xaxis_tickangle=-20, showlegend=False)
    st.plotly_chart(fig_count, use_container_width=True)
with c2:
    fig_pct = px.bar(
        count_df, x="Nama_Segmen", y="Persentase", color="Nama_Segmen",
        color_discrete_map=seg_color_map, text="Persentase",
        title="Persentase Pelanggan per Segmen",
        labels={"Nama_Segmen": "Segmen", "Persentase": "Persentase (%)"},
    )
    fig_pct.update_layout(xaxis_tickangle=-20, showlegend=False)
    st.plotly_chart(fig_pct, use_container_width=True)

st.dataframe(count_df.rename(columns=DISPLAY_LABELS), use_container_width=True, hide_index=True)

# Tabel data berlabel klaster yang bisa difilter
st.subheader("Tabel Data Pelanggan dengan Label Klaster")
filter_seg = st.multiselect(
    "Filter segmen (kosongkan untuk semua)",
    options=sorted(df_labeled["Nama_Segmen"].dropna().unique().tolist()),
    default=[],
)
table_df = df_labeled.copy()
if filter_seg:
    table_df = table_df[table_df["Nama_Segmen"].isin(filter_seg)]
st.caption(f"Menampilkan {len(table_df)} dari {len(df_labeled)} baris data kerja.")
st.dataframe(rename_for_display(table_df), use_container_width=True)
