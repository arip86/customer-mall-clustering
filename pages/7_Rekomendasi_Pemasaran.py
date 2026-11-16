"""Halaman 7 - Rekomendasi Pemasaran (Rumusan Masalah 4 / Tujuan 4)"""

import streamlit as st

from utils import (
    ALIAS_DISPLAY,
    DEFAULT_FEATURES,
    DEFAULT_K,
    ensure_cluster_results,
    generate_marketing_recommendations,
    inject_custom_css,
    render_global_sidebar,
    render_page_header,
)

st.set_page_config(page_title="Rekomendasi Pemasaran", page_icon="🎯", layout="wide")
inject_custom_css()
render_global_sidebar()

render_page_header(
    "7. Rekomendasi Pemasaran",
    "Menjawab Rumusan Masalah 4 / Tujuan 4: strategi pemasaran berbasis profil tiap segmen.",
)

df_work = st.session_state.get("df_work")
if df_work is None or df_work.empty:
    st.error("Data kerja tidak tersedia atau kosong. Periksa sumber data dan filter usia di sidebar.")
    st.stop()

ok, err = ensure_cluster_results()
if not ok:
    st.error(err or "Gagal menyiapkan hasil klaster untuk rekomendasi.")
    st.info("Buka Halaman 4 untuk memilih fitur dan k final, lalu kembali ke halaman ini.")
    st.stop()

cluster_summary = st.session_state.get("cluster_summary")
df_labeled = st.session_state.get("df_labeled")
segment_names = st.session_state.get("segment_names", {})
seg_color_map = st.session_state.get("seg_color_map", {})
selected_features = st.session_state.get("selected_features", DEFAULT_FEATURES)
k_final = int(st.session_state.get("k_final", DEFAULT_K))

if cluster_summary is None or df_labeled is None or cluster_summary.empty:
    st.error("Ringkasan klaster belum tersedia. Silakan periksa pengaturan klaster di Halaman 4.")
    st.stop()

overall_income = float(df_labeled["Annual_Income($)"].mean()) if "Annual_Income($)" in df_labeled.columns else 0.0
overall_spending = float(df_labeled["Spending Score(1-100)"].mean()) if "Spending Score(1-100)" in df_labeled.columns else 0.0

st.markdown(
    f"<div class='soft-info'>Rekomendasi dihitung dari <b>data kerja aktif</b> "
    f"(k = {k_final}, fitur: {', '.join([ALIAS_DISPLAY.get(a, a) for a in selected_features])}). "
    f"Rata-rata keseluruhan — Pendapatan: <b>{overall_income:,.0f}</b>, Skor Belanja: <b>{overall_spending:.2f}</b>. "
    f"Total segmen ditampilkan: <b>{len(cluster_summary)}</b>.</div>",
    unsafe_allow_html=True,
)
st.write("")

# Render kartu per segmen (pastikan semua segmen tampil)
sorted_summary = cluster_summary.sort_values("Cluster")
for _, row in sorted_summary.iterrows():
    cid = int(row["Cluster"])
    seg_name = row.get("Nama_Segmen", segment_names.get(cid, f"Klaster {cid}"))
    inc = float(row.get("Annual_Income($)", 0))
    spd = float(row.get("Spending Score(1-100)", 0))
    fam = float(row.get("Family_Size", 0))
    age_m = float(row.get("Age", 0))
    work_m = float(row.get("Work_Experience", 0))
    cnt = int(row.get("Jumlah", 0))
    pct = float(row.get("Persentase", 0))
    dom_prof = str(row.get("Profesi_Dominan", "-"))
    border_color = (seg_color_map or {}).get(seg_name, "#2563EB")

    recs = generate_marketing_recommendations(
        income_mean=inc,
        spending_mean=spd,
        family_size_mean=fam,
        overall_income=overall_income,
        overall_spending=overall_spending,
        dominant_profession=dom_prof,
    )

    with st.container(border=True):
        st.markdown(
            f"""
<div style="border-left:6px solid {border_color}; padding:0.2rem 0 0.2rem 0.8rem; margin-bottom:0.4rem;">
  <div style="font-size:1.15rem; font-weight:700; color:#1E3A8A;">Klaster {cid}: {seg_name}</div>
  <div style="color:#3B82F6; font-size:0.95rem;">{cnt} pelanggan ({pct}%) dari data kerja • Profesi dominan: <b>{dom_prof}</b></div>
</div>
""",
            unsafe_allow_html=True,
        )
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Rata-rata Usia", f"{age_m:.1f}")
        m2.metric("Rata-rata Pendapatan", f"{inc:,.0f}")
        m3.metric("Rata-rata Skor Belanja", f"{spd:.1f}")
        m4.metric("Rata-rata Pengalaman Kerja", f"{work_m:.1f}")
        m5.metric("Rata-rata Ukuran Keluarga", f"{fam:.1f}")

        st.markdown("**Dasar & rekomendasi strategi pemasaran (berbasis aturan):**")
        for r in recs:
            st.markdown(f"- {r}")

st.info(
    "Aturan utama: (1) Pendapatan tinggi & belanja tinggi → loyalitas premium; "
    "(2) Pendapatan tinggi & belanja rendah → promosi pemicu minat belanja; "
    "(3) Pendapatan rendah & belanja tinggi → diskon/cicilan; "
    "(4) Pendapatan rendah & belanja rendah → promosi hemat. "
    "Tambahan: paket keluarga bila ukuran keluarga tinggi dan penawaran khusus untuk profesi dominan."
)
