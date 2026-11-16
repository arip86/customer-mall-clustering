"""Halaman 6 - Profil Segmen (Rumusan Masalah 3 / Tujuan 3)"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (
    ALIAS_DISPLAY,
    DEFAULT_FEATURES,
    DEFAULT_K,
    DISPLAY_LABELS,
    RANDOM_STATE,
    build_cluster_profile,
    ensure_cluster_results,
    get_feature_columns,
    inject_custom_css,
    render_global_sidebar,
    render_page_header,
)

st.set_page_config(page_title="Profil Segmen", page_icon="👥", layout="wide")
inject_custom_css()
render_global_sidebar()

render_page_header("6. Profil Segmen", "Menjawab Rumusan Masalah 3 / Tujuan 3: karakteristik, komposisi, dan penamaan segmen.")

df_work = st.session_state.get("df_work")
if df_work is None or df_work.empty:
    st.error("Data kerja tidak tersedia atau kosong. Periksa sumber data dan filter usia di sidebar.")
    st.stop()

ok, err = ensure_cluster_results()
if not ok:
    st.error(err or "Gagal menyiapkan profil segmen.")
    st.info("Buka Halaman 4 untuk memilih fitur dan k final, lalu kembali ke halaman ini.")
    st.stop()

df_labeled = st.session_state.get("df_labeled")
cluster_summary = st.session_state.get("cluster_summary")
segment_names = st.session_state.get("segment_names", {})
seg_color_map = st.session_state.get("seg_color_map", {})
selected_features = st.session_state.get("selected_features", DEFAULT_FEATURES)
k_final = int(st.session_state.get("k_final", DEFAULT_K))

# Komposisi gender/profesi
_, _, gender_comp, prof_comp = build_cluster_profile(df_labeled, "Cluster")
st.session_state["gender_comp"] = gender_comp
st.session_state["prof_comp"] = prof_comp

attr_cols = [c for c in ["Age", "Annual_Income($)", "Spending Score(1-100)", "Work_Experience", "Family_Size"] if c in df_labeled.columns]
overall_means = df_labeled[attr_cols].mean()

st.caption(f"Konfigurasi: k={k_final}, fitur: {', '.join([ALIAS_DISPLAY.get(a,a) for a in selected_features])}")

# Tabel rata-rata vs keseluruhan
st.subheader("Rata-rata Atribut per Klaster vs Rata-rata Keseluruhan")
profile_tbl = cluster_summary[["Cluster", "Nama_Segmen", "Jumlah", "Persentase", "Profesi_Dominan"] + attr_cols].copy()
display_tbl = profile_tbl.copy()
for col in attr_cols:
    display_tbl[f"Delta {DISPLAY_LABELS.get(col,col)}"] = (display_tbl[col] - overall_means[col]).round(2)
rename_map = {c: DISPLAY_LABELS.get(c, c) for c in attr_cols}
display_tbl = display_tbl.rename(columns=rename_map)
st.dataframe(display_tbl.sort_values("Cluster"), use_container_width=True, hide_index=True)

o1, o2 = st.columns([1, 2])
with o1:
    st.markdown("**Rata-rata keseluruhan (data kerja):**")
    st.dataframe(overall_means.to_frame("Rata-rata").rename(index=DISPLAY_LABELS), use_container_width=True)

with o2:
    st.markdown("**Heatmap Perbandingan Rata-rata Atribut per Segmen**")
    heat = cluster_summary.set_index("Nama_Segmen")[attr_cols].copy()
    heat_ratio = heat.copy()
    for col in attr_cols:
        denom = overall_means[col] if overall_means[col] != 0 else 1
        heat_ratio[col] = (heat[col] / denom)
    heat_ratio.columns = [DISPLAY_LABELS.get(c, c) for c in heat_ratio.columns]
    fig_heat = px.imshow(
        heat_ratio, text_auto=".2f", aspect="auto", color_continuous_scale="Blues",
        title="Rasio Rata-rata Segmen / Rata-rata Keseluruhan (1.00 = sama dengan keseluruhan)",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# Radar chart
st.subheader("Radar Chart Profil Segmen")
radar_df = cluster_summary[["Nama_Segmen"] + attr_cols].copy()
norm = radar_df.copy()
for col in attr_cols:
    cmin, cmax = radar_df[col].min(), radar_df[col].max()
    norm[col] = 0.5 if (cmax - cmin == 0) else (radar_df[col] - cmin) / (cmax - cmin)
categories = [DISPLAY_LABELS.get(c, c) for c in attr_cols]
fig_radar = go.Figure()
for _, row in norm.iterrows():
    seg = row["Nama_Segmen"]
    values = [row[c] for c in attr_cols] + [row[attr_cols[0]]]
    cats = categories + [categories[0]]
    fig_radar.add_trace(go.Scatterpolar(
        r=values, theta=cats, fill='toself', name=seg,
        line=dict(color=seg_color_map.get(seg)) if seg_color_map else None,
    ))
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    showlegend=True, title="Profil Relatif Segmen (skala ternormalisasi 0-1 antar segmen)",
)
st.plotly_chart(fig_radar, use_container_width=True)

# Komposisi Gender
st.subheader("Komposisi Gender per Klaster")
if not gender_comp.empty:
    g_disp = gender_comp.copy().rename(columns={"Gender": "Jenis Kelamin"})
    g1, g2 = st.columns(2)
    with g1:
        st.dataframe(
            g_disp[["Cluster", "Nama_Segmen", "Jenis Kelamin", "Jumlah", "Persentase"]].sort_values(["Cluster", "Jenis Kelamin"]),
            use_container_width=True, hide_index=True,
        )
    with g2:
        fig_g = px.bar(
            g_disp, x="Nama_Segmen", y="Jumlah", color="Jenis Kelamin", barmode="stack",
            title="Komposisi Gender per Segmen",
            labels={"Nama_Segmen": "Segmen", "Jumlah": "Jumlah"},
        )
        fig_g.update_layout(xaxis_tickangle=-20)
        st.plotly_chart(fig_g, use_container_width=True)
else:
    st.info("Data gender tidak tersedia.")

# Komposisi Profesi
st.subheader("Komposisi Profesi per Klaster")
if not prof_comp.empty:
    p_disp = prof_comp.copy().rename(columns={"Profession": "Profesi"})
    p1, p2 = st.columns(2)
    with p1:
        st.dataframe(
            p_disp[["Cluster", "Nama_Segmen", "Profesi", "Jumlah", "Persentase"]].sort_values(["Cluster", "Jumlah"], ascending=[True, False]),
            use_container_width=True, hide_index=True,
        )
    with p2:
        fig_p = px.bar(
            prof_comp, x="Nama_Segmen", y="Jumlah", color="Profession", barmode="stack",
            title="Komposisi Profesi per Segmen",
            labels={"Nama_Segmen": "Segmen", "Jumlah": "Jumlah", "Profession": "Profesi"},
        )
        fig_p.update_layout(xaxis_tickangle=-20)
        st.plotly_chart(fig_p, use_container_width=True)
else:
    st.info("Data profesi tidak tersedia.")

# Nama deskriptif otomatis
st.subheader("Nama Deskriptif Otomatis Segmen")
st.markdown(
    "Nama segmen ditentukan otomatis dari posisi **rata-rata Pendapatan Tahunan** dan **Skor Belanja** klaster "
    "relatif terhadap rata-rata keseluruhan (rendah/sedang/tinggi)."
)
name_tbl = cluster_summary[["Cluster", "Nama_Segmen", "Annual_Income($)", "Spending Score(1-100)", "Jumlah", "Persentase"]].copy()
name_tbl = name_tbl.rename(columns={
    "Annual_Income($)": "Rata-rata Pendapatan", "Spending Score(1-100)": "Rata-rata Skor Belanja",
    "Nama_Segmen": "Nama Segmen", "Persentase": "Persentase (%)",
})
st.dataframe(name_tbl.sort_values("Cluster"), use_container_width=True, hide_index=True)
st.caption(
    "Catatan: Karena pemisahan klaster cenderung lemah (Silhouette rendah), nama dan profil ini merupakan "
    "pengelompokan praktis untuk kebutuhan pemasaran."
)
