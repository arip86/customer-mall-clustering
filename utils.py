"""
Fungsi utilitas untuk aplikasi laporan interaktif segmentasi pelanggan mall (K-Means).
Mencakup: pemuatan/validasi data, prapemrosesan, kualitas data, filter, caching,
evaluasi klaster, penamaan segmen, profil, rekomendasi, prediksi, ringkasan otomatis,
serta UI helper (sidebar, logo, CSS, state global).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import plotly.express as px
    import plotly.io as pio

    pio.templates.default = "plotly_white"
except Exception:  # pragma: no cover - fallback palet jika plotly belum terpasang
    class _DummyPX:
        class colors:
            qualitative = type(
                "Q",
                (),
                {
                    "Plotly": [
                        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
                        "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
                    ],
                    "Bold": [
                        "#7F3C8D", "#11A579", "#3969AC", "#F2B701", "#E73F74",
                        "#80BA5A", "#E68310", "#008695", "#CF1C90", "#f97b72",
                    ],
                    "Safe": [
                        "#88CCEE", "#CC6677", "#DDCC77", "#117733", "#332288",
                        "#AA4499", "#44AA99", "#999933", "#882255", "#661100",
                    ],
                },
            )()

    px = _DummyPX()

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    import streamlit as st
except Exception:  # pragma: no cover
    class _DummyStreamlit:
        session_state: Dict[str, Any] = {}

        def cache_data(self, *args, **kwargs):
            def decorator(func):
                return func

            if args and callable(args[0]) and len(args) == 1 and not kwargs:
                return args[0]
            return decorator

        def sidebar(self):
            return self

        def markdown(self, *a, **k):
            return None

        def image(self, *a, **k):
            return None

        def caption(self, *a, **k):
            return None

        def header(self, *a, **k):
            return None

        def subheader(self, *a, **k):
            return None

        def radio(self, *a, **k):
            return a[1][0] if len(a) > 1 else None

        def file_uploader(self, *a, **k):
            return None

        def checkbox(self, *a, **k):
            return k.get("value", False)

        def slider(self, *a, **k):
            return k.get("value")

        def expander(self, *a, **k):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def warning(self, *a, **k):
            return None

        def info(self, *a, **k):
            return None

        def error(self, *a, **k):
            return None

        def success(self, *a, **k):
            return None

        def divider(self, *a, **k):
            return None

        def logo(self, *a, **k):
            return None

    st = _DummyStreamlit()


# -----------------------------
# Konstanta & konfigurasi
# -----------------------------
REQUIRED_COLUMNS = [
    "CustomerID",
    "Gender",
    "Age",
    "Annual_Income($)",
    "Spending Score(1-100)",
    "Profession",
    "Work_Experience",
    "Family_Size",
]

DISPLAY_LABELS = {
    "CustomerID": "ID Pelanggan",
    "Gender": "Jenis Kelamin",
    "Age": "Usia",
    "Annual_Income($)": "Pendapatan Tahunan ($)",
    "Spending Score(1-100)": "Skor Belanja (1-100)",
    "Profession": "Profesi",
    "Work_Experience": "Pengalaman Kerja (tahun)",
    "Family_Size": "Ukuran Keluarga",
    "Cluster": "Klaster",
    "Nama_Segmen": "Nama Segmen",
    "Jumlah": "Jumlah",
    "Persentase": "Persentase (%)",
}

FEATURE_ALIAS = {
    "Age": "Age",
    "Income": "Annual_Income($)",
    "Spending": "Spending Score(1-100)",
    "Work_Experience": "Work_Experience",
    "Family_Size": "Family_Size",
}

ALIAS_DISPLAY = {
    "Age": "Usia",
    "Income": "Pendapatan Tahunan ($)",
    "Spending": "Skor Belanja (1-100)",
    "Work_Experience": "Pengalaman Kerja (tahun)",
    "Family_Size": "Ukuran Keluarga",
}

NUMERIC_FEATURE_OPTIONS = ["Age", "Income", "Spending", "Work_Experience", "Family_Size"]
DEFAULT_FEATURES = ["Age", "Income", "Spending", "Work_Experience", "Family_Size"]
DEFAULT_K = 5
RANDOM_STATE = 42

CLUSTER_FORMING_COLS = [
    "Age",
    "Annual_Income($)",
    "Spending Score(1-100)",
    "Work_Experience",
    "Family_Size",
]
PROFILE_COMPLEMENT_COLS = ["CustomerID", "Gender", "Profession"]

APP_DIR = Path(__file__).resolve().parent
LOGO_CANDIDATES = ["logo_unpak.png", "images (1).png", "logo.png", "unpak.png"]


# -----------------------------
# UI helper (logo, CSS, header)
# -----------------------------
def get_logo_path() -> Optional[Path]:
    """Mencari path logo UNPAK yang tersedia di folder proyek."""
    for name in LOGO_CANDIDATES:
        p = APP_DIR / name
        if p.exists():
            return p
    return None


def inject_custom_css() -> None:
    """Menyisipkan CSS kustom bertema dashboard putih dengan palet biru (tanpa hitam)."""
    st.markdown(
        """
        <style>
            :root {
                --blue-primary: #2563EB;
                --blue-primary-dark: #1D4ED8;
                --blue-dark: #1E3A8A;
                --blue-medium: #1D4ED8;
                --blue-muted: #3B82F6;
                --blue-soft: #60A5FA;
                --blue-border: #BFDBFE;
                --blue-surface: #F8FAFF;
                --blue-surface-2: #EFF6FF;
            }
            html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .stApp {
                background-color: #ffffff !important;
                color: var(--blue-dark) !important;
            }
            [data-testid="stHeader"] {
                background: rgba(255,255,255,0.95);
            }
            .block-container {
                padding-top: 1rem;
                padding-bottom: 2.5rem;
                max-width: 1320px;
            }
            h1, h2, h3, h4, h5, h6, p, span, label, li, div, a {
                color: var(--blue-dark);
            }
            h1, h2, h3 { font-weight: 600; }
            h2 { font-size: 1.35rem; margin-top: 0.5rem; }
            h3 { font-size: 1.1rem; }
            a { color: var(--blue-primary); }

            /* Sidebar - light blue-white style */
            [data-testid="stSidebar"] {
                background-color: #ffffff !important;
                border-right: 1px solid var(--blue-border);
            }
            [data-testid="stSidebar"] > div:first-child {
                background-color: #ffffff;
                padding-top: 1rem;
            }
            [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] div, [data-testid="stSidebar"] small {
                color: var(--blue-dark) !important;
            }
            [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
            [data-testid="stSidebar"] h4, [data-testid="stSidebar"] strong {
                color: var(--blue-primary) !important;
            }
            [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
                color: var(--blue-muted) !important;
            }
            [data-testid="stSidebar"] [data-testid="stExpander"] {
                background: var(--blue-surface);
                border: 1px solid var(--blue-border);
                border-radius: 0.5rem;
            }
            [data-testid="stSidebar"] hr { border-color: var(--blue-border); }
            [data-testid="stSidebarNav"] a {
                border-radius: 0.375rem;
                margin-bottom: 2px;
                color: var(--blue-dark) !important;
            }
            [data-testid="stSidebarNav"] a span { color: var(--blue-dark) !important; }
            [data-testid="stSidebarNav"] a:hover {
                background-color: var(--blue-surface-2) !important;
                color: var(--blue-primary) !important;
            }
            [data-testid="stSidebarNav"] a[aria-current="page"] {
                background-color: #DBEAFE !important;
                color: var(--blue-primary-dark) !important;
                font-weight: 600;
            }
            [data-testid="stSidebarNav"] a[aria-current="page"] span {
                color: var(--blue-primary-dark) !important;
            }

            /* Page header / dashboard banner */
            .app-header {
                background: #ffffff;
                border: 1px solid var(--blue-border);
                border-left: 6px solid var(--blue-primary);
                border-radius: 0.5rem;
                padding: 1rem 1.25rem;
                margin-bottom: 1.25rem;
                box-shadow: 0 0.25rem 0.75rem rgba(37, 99, 235, 0.08);
            }
            .app-header h1 {
                margin: 0;
                font-size: 1.55rem;
                font-weight: 700;
                color: var(--blue-primary-dark);
            }
            .app-header p {
                margin: 0.3rem 0 0 0;
                color: var(--blue-muted);
                font-size: 0.98rem;
            }
            .app-header-accent { color: var(--blue-primary); }

            /* Cards */
            .section-card {
                background: #ffffff;
                border: 1px solid var(--blue-border);
                border-radius: 0.5rem;
                padding: 1rem 1.1rem;
                margin-bottom: 1rem;
                box-shadow: 0 0.2rem 0.5rem rgba(37, 99, 235, 0.06);
            }
            .soft-info {
                background: var(--blue-surface-2);
                border: 1px solid var(--blue-border);
                border-left: 4px solid var(--blue-primary);
                border-radius: 0.5rem;
                padding: 0.85rem 1rem;
                color: var(--blue-dark);
            }
            div[data-testid="stVerticalBlockBorderWrapper"] {
                background: #ffffff;
                border: 1px solid var(--blue-border) !important;
                border-radius: 0.5rem !important;
                box-shadow: 0 0.2rem 0.5rem rgba(37, 99, 235, 0.06);
                padding: 0.25rem 0.5rem;
            }
            div[data-testid="stVerticalBlockBorderWrapper"] > div { background: transparent; }

            /* KPI / metrics */
            div[data-testid="stMetric"] {
                background: #ffffff;
                border: 1px solid var(--blue-border);
                border-top: 3px solid var(--blue-primary);
                border-radius: 0.5rem;
                padding: 0.75rem 1rem;
                box-shadow: 0 0.2rem 0.5rem rgba(37, 99, 235, 0.06);
            }
            div[data-testid="stMetric"] label,
            div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
                color: var(--blue-muted) !important;
                font-size: 0.85rem;
                font-weight: 500;
            }
            div[data-testid="stMetric"] [data-testid="stMetricValue"] {
                color: var(--blue-primary-dark) !important;
                font-weight: 700;
            }
            div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
                color: var(--blue-medium) !important;
            }

            /* Tables & charts */
            div[data-testid="stDataFrame"] {
                background: #ffffff;
                border: 1px solid var(--blue-border);
                border-radius: 0.5rem;
                overflow: hidden;
                box-shadow: 0 0.2rem 0.5rem rgba(37, 99, 235, 0.05);
            }
            .stPlotlyChart {
                background: #ffffff;
                border: 1px solid var(--blue-border);
                border-radius: 0.5rem;
                padding: 0.35rem;
                box-shadow: 0 0.2rem 0.5rem rgba(37, 99, 235, 0.05);
                margin-bottom: 0.5rem;
            }

            /* Inputs & buttons */
            .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
                background-color: var(--blue-primary);
                color: #ffffff !important;
                border: 1px solid var(--blue-primary);
                border-radius: 0.375rem;
                font-weight: 500;
            }
            .stButton > button, .stDownloadButton > button {
                border-radius: 0.375rem;
                border: 1px solid var(--blue-border);
                color: var(--blue-dark);
            }
            .stButton > button:hover, .stDownloadButton > button:hover {
                border-color: var(--blue-primary);
                color: var(--blue-primary);
            }

            /* Alerts softer blue */
            div[data-testid="stAlert"] {
                border-radius: 0.5rem;
                border: 1px solid var(--blue-border);
            }

            /* Captions */
            .stCaption, small, [data-testid="stCaptionContainer"] {
                color: var(--blue-muted) !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, subtitle: str = "") -> None:
    """Merender header halaman dengan logo UNPAK jika tersedia."""
    logo_path = get_logo_path()
    if logo_path and logo_path.exists():
        col_logo, col_title = st.columns([1, 11], vertical_alignment="center")
        with col_logo:
            try:
                st.image(str(logo_path), use_container_width=True)
            except Exception:
                pass
        with col_title:
            st.markdown(
                f"""<div class="app-header"><h1>{title}</h1><p>{subtitle}</p></div>""",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f"""<div class="app-header"><h1>{title}</h1><p>{subtitle}</p></div>""",
            unsafe_allow_html=True,
        )


def apply_plotly_style(fig):
    """Terapkan gaya plotly putih-biru agar konsisten dengan tema dashboard."""
    try:
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="#ffffff",
            font=dict(color="#1E3A8A"),
            title_font=dict(color="#1D4ED8"),
            margin=dict(l=20, r=20, t=55, b=40),
            legend=dict(font=dict(color="#1E3A8A"), bgcolor="rgba(248,250,255,0.85)"),
            xaxis=dict(title_font=dict(color="#1D4ED8"), tickfont=dict(color="#1E3A8A"), gridcolor="#DBEAFE", zerolinecolor="#BFDBFE"),
            yaxis=dict(title_font=dict(color="#1D4ED8"), tickfont=dict(color="#1E3A8A"), gridcolor="#DBEAFE", zerolinecolor="#BFDBFE"),
        )
    except Exception:
        pass
    return fig


# -----------------------------
# State & palet
# -----------------------------
def init_session_state() -> None:
    """Inisialisasi nilai default session_state."""
    defaults = {
        "k_final": DEFAULT_K,
        "random_state": RANDOM_STATE,
        "selected_features": DEFAULT_FEATURES.copy(),
        "use_age_filter": False,
        "age_range": None,
        "data_source": "default",
        "upload_error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_cluster_color_map(cluster_ids: List[int]) -> Dict[int, str]:
    """Menghasilkan pemetaan warna deterministik untuk setiap klaster."""
    palette = px.colors.qualitative.Plotly + px.colors.qualitative.Bold + px.colors.qualitative.Safe
    sorted_ids = sorted([int(c) for c in cluster_ids])
    color_map: Dict[int, str] = {}
    for i, cid in enumerate(sorted_ids):
        color_map[cid] = palette[i % len(palette)]
    return color_map


def get_segment_color_map(labels_series: pd.Series, segment_names: Dict[int, str]) -> Dict[str, str]:
    """Pemetaan warna berdasarkan nama segmen (konsisten per cluster id)."""
    cluster_ids = sorted(pd.Series(labels_series).astype(int).unique().tolist())
    cmap = get_cluster_color_map(cluster_ids)
    return {segment_names.get(cid, f"Klaster {cid}"): cmap[cid] for cid in cluster_ids}


# -----------------------------
# Pemuatan & validasi data
# -----------------------------
@st.cache_data(show_spinner=False)
def load_default_data(path: str = "Customer_mall.csv") -> pd.DataFrame:
    """Memuat data default dari file CSV di repo."""
    csv_path = APP_DIR / path
    if not csv_path.exists():
        csv_path = Path(path)
    return pd.read_csv(csv_path)


@st.cache_data(show_spinner=False)
def load_uploaded_data(file_bytes: bytes) -> pd.DataFrame:
    """Memuat data dari file CSV yang diunggah (berbasis bytes agar cache stabil)."""
    import io
    return pd.read_csv(io.BytesIO(file_bytes))


def validate_columns(df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
    """Memvalidasi keberadaan kolom wajib pada DataFrame."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return len(missing) == 0, missing, list(df.columns)


# -----------------------------
# Prapemrosesan & kualitas data
# -----------------------------
def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Membersihkan dan menyiapkan data tanpa menghapus baris secara diam-diam."""
    data = df.copy()
    warnings: Dict[str, Any] = {}

    # Profession kosong/NaN -> Unknown
    if "Profession" in data.columns:
        prof_missing = int(data["Profession"].isna().sum())
        prof_blank = int(data["Profession"].astype(str).str.strip().eq("").sum())
        warnings["profession_missing_before"] = prof_missing
        warnings["profession_blank_before"] = prof_blank
        data["Profession"] = data["Profession"].fillna("Unknown")
        data.loc[data["Profession"].astype(str).str.strip().eq(""), "Profession"] = "Unknown"
        warnings["profession_filled_unknown"] = prof_missing + prof_blank
    else:
        warnings["profession_missing_before"] = 0
        warnings["profession_blank_before"] = 0
        warnings["profession_filled_unknown"] = 0

    # Konversi numerik aman
    numeric_cols = [
        "CustomerID",
        "Age",
        "Annual_Income($)",
        "Spending Score(1-100)",
        "Work_Experience",
        "Family_Size",
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    warnings["duplicates"] = int(data.duplicated().sum())
    warnings["rows_total"] = int(len(data))
    warnings["cols_total"] = int(len(data.columns))

    if "Age" in data.columns:
        age_series = data["Age"].dropna()
        warnings["age_min"] = float(age_series.min()) if not age_series.empty else None
        warnings["age_max"] = float(age_series.max()) if not age_series.empty else None
        warnings["age_zero_count"] = int((data["Age"] == 0).sum())
        warnings["age_unusual_count"] = int(((data["Age"] < 18) | (data["Age"] > 80)).sum())
    else:
        warnings["age_min"] = warnings["age_max"] = None
        warnings["age_zero_count"] = warnings["age_unusual_count"] = 0

    if "Annual_Income($)" in data.columns:
        warnings["income_zero_count"] = int((data["Annual_Income($)"] == 0).sum())
    else:
        warnings["income_zero_count"] = 0

    if "Spending Score(1-100)" in data.columns:
        sp = data["Spending Score(1-100)"].dropna()
        warnings["spending_min"] = float(sp.min()) if not sp.empty else None
        warnings["spending_max"] = float(sp.max()) if not sp.empty else None
    else:
        warnings["spending_min"] = warnings["spending_max"] = None

    warnings["missing_by_col"] = data.isna().sum().to_dict()
    return data, warnings


def compute_data_quality_messages(warnings: Dict[str, Any]) -> List[str]:
    """Menyusun pesan peringatan kualitas data."""
    messages = []
    filled = warnings.get("profession_filled_unknown", 0)
    messages.append(
        f"Profesi kosong/NaN: {filled} baris diisi dengan \"Unknown\" (baris tidak dihapus)."
    )
    messages.append(
        "Usia berkisar "
        f"{warnings.get('age_min')}–{warnings.get('age_max')} dengan sebaran sangat lebar; "
        f"nilai tidak lazim (termasuk 0 atau usia sangat tua) tidak dihapus otomatis. "
        f"Jumlah Age=0: {warnings.get('age_zero_count', 0)}, "
        f"jumlah di luar rentang lazim (<18 atau >80): {warnings.get('age_unusual_count', 0)}."
    )
    messages.append(
        f"Pendapatan Tahunan bernilai 0 pada {warnings.get('income_zero_count', 0)} baris (ditandai sebagai peringatan)."
    )
    messages.append(
        "Korelasi antar variabel numerik mendekati nol dan Silhouette Score cenderung rendah "
        "(sekitar 0,15–0,17). Segmen bersifat pengelompokan praktis, bukan kelompok alami yang terpisah tegas."
    )
    messages.append(f"Jumlah baris duplikat: {warnings.get('duplicates', 0)}.")
    return messages


def filter_by_age(df: pd.DataFrame, min_age: int, max_age: int) -> pd.DataFrame:
    """Memfilter data berdasarkan rentang usia pada salinan kerja."""
    if "Age" not in df.columns:
        return df.copy()
    return df[(df["Age"] >= min_age) & (df["Age"] <= max_age)].copy()


# -----------------------------
# Sidebar & state global (dipakai semua halaman)
# -----------------------------
def ensure_app_state() -> None:
    """Memastikan data dan pengaturan global tersedia di session_state untuk semua halaman."""
    init_session_state()
    st.session_state.setdefault("upload_error", None)

    # Muat/simpan sumber data di state jika belum ada atau sumber berubah melalui kontrol lokal
    raw_df = st.session_state.get("raw_df")
    processed_df = st.session_state.get("processed_df")
    quality_warnings = st.session_state.get("quality_warnings")

    # Jika belum ada data sama sekali, coba muat default
    if raw_df is None or processed_df is None or quality_warnings is None:
        try:
            raw_df = load_default_data("Customer_mall.csv")
            is_valid, missing_cols, available_cols = validate_columns(raw_df)
            if not is_valid:
                st.session_state["upload_error"] = (
                    "Kolom wajib tidak ditemukan pada data default. "
                    f"Hilang: {', '.join(missing_cols)}. Tersedia: {', '.join(available_cols)}."
                )
                st.session_state["raw_df"] = None
                st.session_state["processed_df"] = None
                st.session_state["quality_warnings"] = None
            else:
                processed_df, quality_warnings = preprocess_data(raw_df)
                st.session_state["raw_df"] = raw_df
                st.session_state["processed_df"] = processed_df
                st.session_state["quality_warnings"] = quality_warnings
                st.session_state["upload_error"] = None
        except Exception as e:
            st.session_state["upload_error"] = f"Gagal memuat data default: {e}"
            st.session_state["raw_df"] = None
            st.session_state["processed_df"] = None
            st.session_state["quality_warnings"] = None

    # Terapkan filter usia ke df_work bila data tersedia
    processed_df = st.session_state.get("processed_df")
    if processed_df is not None:
        age_min_data = int(processed_df["Age"].min()) if "Age" in processed_df.columns and not processed_df["Age"].dropna().empty else 0
        age_max_data = int(processed_df["Age"].max()) if "Age" in processed_df.columns and not processed_df["Age"].dropna().empty else 100
        if not st.session_state.get("age_range"):
            st.session_state["age_range"] = (age_min_data, age_max_data)
        if st.session_state.get("use_age_filter"):
            ar = st.session_state.get("age_range") or (age_min_data, age_max_data)
            ar = (max(age_min_data, int(ar[0])), min(age_max_data, int(ar[1])))
            st.session_state["age_range"] = ar
            st.session_state["df_work"] = filter_by_age(processed_df, ar[0], ar[1])
        else:
            st.session_state["df_work"] = processed_df.copy()
    else:
        st.session_state["df_work"] = None


def render_global_sidebar(show_cluster_controls: bool = False) -> None:
    """Merender sidebar global (logo, sumber data, filter, status) di semua halaman."""
    init_session_state()
    ensure_app_state()

    # Logo & branding
    logo_path = get_logo_path()
    if logo_path and logo_path.exists():
        try:
            if hasattr(st, "logo"):
                st.logo(str(logo_path), icon_image=str(logo_path))
        except Exception:
            pass
        st.sidebar.markdown(
            "<div style='text-align:center; padding:0.4rem 0 0.6rem 0; border-bottom:1px solid #BFDBFE; margin-bottom:0.8rem;'>",
            unsafe_allow_html=True,
        )
        st.sidebar.image(str(logo_path), use_container_width=True)
        st.sidebar.markdown(
            "<div style='text-align:center; font-weight:700; color:#2563EB; margin-top:0.2rem;'>Universitas Pakuan</div>"
            "<div style='text-align:center; font-size:0.85rem; color:#3B82F6; margin-bottom:0.2rem;'>Dashboard Segmentasi Pelanggan</div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            "<div style='font-weight:700; color:#2563EB; font-size:1.05rem; margin-bottom:0.2rem;'>Dashboard Segmentasi</div>"
            "<div style='font-size:0.85rem; color:#3B82F6; border-bottom:1px solid #BFDBFE; padding-bottom:0.6rem; margin-bottom:0.6rem;'>Universitas Pakuan</div>",
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("#### Pengaturan Global")

    # Sumber data
    st.sidebar.subheader("Sumber Data")
    source_options = ["Gunakan data default (Customer_mall.csv)", "Upload CSV sendiri"]
    current_source = st.session_state.get("data_source", "default")
    default_index = 0 if current_source == "default" else 1
    data_source = st.sidebar.radio("Pilih sumber data", options=source_options, index=default_index)

    if data_source == "Upload CSV sendiri":
        st.session_state["data_source"] = "upload"
        uploaded_file = st.sidebar.file_uploader("Unggah file CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                raw_df = load_uploaded_data(uploaded_file.getvalue())
                is_valid, missing_cols, available_cols = validate_columns(raw_df)
                if not is_valid:
                    st.session_state["upload_error"] = (
                        "Kolom wajib tidak ditemukan pada CSV. "
                        f"Kolom hilang: {', '.join(missing_cols)}. "
                        f"Kolom tersedia: {', '.join(available_cols)}."
                    )
                else:
                    processed_df, quality_warnings = preprocess_data(raw_df)
                    st.session_state["raw_df"] = raw_df
                    st.session_state["processed_df"] = processed_df
                    st.session_state["quality_warnings"] = quality_warnings
                    st.session_state["upload_error"] = None
                    # reset age range bounds untuk data baru
                    if "Age" in processed_df.columns and not processed_df["Age"].dropna().empty:
                        st.session_state["age_range"] = (int(processed_df["Age"].min()), int(processed_df["Age"].max()))
            except Exception as e:
                st.session_state["upload_error"] = f"Gagal membaca file CSV: {e}"
        else:
            st.sidebar.info("Silakan unggah file CSV untuk melanjutkan.")
    else:
        # jika beralih ke default dan data saat ini bukan default, muat default
        if st.session_state.get("data_source") != "default" or st.session_state.get("raw_df") is None:
            try:
                raw_df = load_default_data("Customer_mall.csv")
                is_valid, missing_cols, available_cols = validate_columns(raw_df)
                if not is_valid:
                    st.session_state["upload_error"] = (
                        "Kolom wajib tidak ditemukan pada data default. "
                        f"Hilang: {', '.join(missing_cols)}. Tersedia: {', '.join(available_cols)}."
                    )
                else:
                    processed_df, quality_warnings = preprocess_data(raw_df)
                    st.session_state["raw_df"] = raw_df
                    st.session_state["processed_df"] = processed_df
                    st.session_state["quality_warnings"] = quality_warnings
                    st.session_state["upload_error"] = None
                    if "Age" in processed_df.columns and not processed_df["Age"].dropna().empty:
                        st.session_state["age_range"] = (int(processed_df["Age"].min()), int(processed_df["Age"].max()))
            except Exception as e:
                st.session_state["upload_error"] = f"Gagal memuat data default: {e}"
        st.session_state["data_source"] = "default"

    # Terapkan status error jika ada
    if st.session_state.get("upload_error"):
        st.sidebar.error(st.session_state["upload_error"])

    # Filter usia
    st.sidebar.subheader("Filter Usia (Opsional)")
    processed_df = st.session_state.get("processed_df")
    if processed_df is not None and "Age" in processed_df.columns and not processed_df["Age"].dropna().empty:
        age_min_data = int(processed_df["Age"].min())
        age_max_data = int(processed_df["Age"].max())
        use_age_filter = st.sidebar.checkbox(
            "Aktifkan filter rentang usia",
            value=bool(st.session_state.get("use_age_filter", False)),
            key="sb_use_age_filter",
        )
        st.session_state["use_age_filter"] = use_age_filter
        current_range = st.session_state.get("age_range") or (age_min_data, age_max_data)
        current_range = (max(age_min_data, int(current_range[0])), min(age_max_data, int(current_range[1])))
        if use_age_filter:
            age_range = st.sidebar.slider(
                "Rentang usia",
                min_value=age_min_data,
                max_value=age_max_data,
                value=current_range,
                step=1,
                key="sb_age_range",
            )
            st.session_state["age_range"] = age_range
            st.session_state["df_work"] = filter_by_age(processed_df, age_range[0], age_range[1])
        else:
            st.session_state["age_range"] = (age_min_data, age_max_data)
            st.session_state["df_work"] = processed_df.copy()
        st.sidebar.caption(
            f"Data kerja: {len(st.session_state.get('df_work', []))} dari {len(processed_df)} baris."
        )
    else:
        st.session_state["df_work"] = processed_df.copy() if processed_df is not None else None
        if processed_df is not None:
            st.sidebar.caption(f"Data kerja: {len(processed_df)} baris.")

    # Info klastering
    st.sidebar.subheader("Pengaturan Klastering")
    st.sidebar.caption(f"Random state tetap: {RANDOM_STATE}")
    st.sidebar.caption(
        f"k final: {st.session_state.get('k_final', DEFAULT_K)} (atur di Halaman 4)"
    )
    feats = st.session_state.get("selected_features", DEFAULT_FEATURES)
    st.sidebar.caption("Fitur: " + ", ".join([ALIAS_DISPLAY.get(f, f) for f in feats]))

    # Peringatan kualitas
    warnings = st.session_state.get("quality_warnings")
    if warnings:
        with st.sidebar.expander("Peringatan Kualitas Data", expanded=False):
            for msg in compute_data_quality_messages(warnings):
                st.warning(msg)


# -----------------------------
# Fitur & klastering
# -----------------------------
def get_feature_columns(selected_aliases: List[str]) -> List[str]:
    """Mengubah alias fitur terpilih menjadi nama kolom asli."""
    return [FEATURE_ALIAS[a] for a in selected_aliases if a in FEATURE_ALIAS]


def prepare_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Menyiapkan DataFrame fitur numerik untuk klastering (imputasi defensif)."""
    feat_df = df[feature_cols].copy()
    for col in feat_df.columns:
        if feat_df[col].isna().any():
            median_val = feat_df[col].median()
            if pd.isna(median_val):
                median_val = 0
            feat_df[col] = feat_df[col].fillna(median_val)
    return feat_df


@st.cache_data(show_spinner=False)
def evaluate_k_range(
    df_features: pd.DataFrame, k_min: int = 2, k_max: int = 10, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """Menghitung inertia dan silhouette score untuk rentang k."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features.values)
    results = []
    n_samples = X_scaled.shape[0]
    for k in range(k_min, k_max + 1):
        if k >= n_samples:
            continue
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertia = float(km.inertia_)
        sil = np.nan
        if len(set(labels)) > 1 and n_samples > k:
            try:
                sil = float(silhouette_score(X_scaled, labels))
            except Exception:
                sil = np.nan
        results.append({"k": k, "inertia": inertia, "silhouette": sil})
    return pd.DataFrame(results)


@st.cache_data(show_spinner=False)
def run_kmeans(df_features: pd.DataFrame, k: int, random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """Menjalankan K-Means dan mengembalikan hasil lengkap."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features.values)
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(X_scaled)
    inertia = float(model.inertia_)
    sil = np.nan
    if len(set(labels)) > 1 and len(labels) > k:
        try:
            sil = float(silhouette_score(X_scaled, labels))
        except Exception:
            sil = np.nan
    pca = PCA(n_components=2, random_state=random_state)
    pca_coords = pca.fit_transform(X_scaled)
    return {
        "model": model,
        "scaler": scaler,
        "labels": labels,
        "inertia": inertia,
        "silhouette": sil,
        "X_scaled": X_scaled,
        "pca_coords": pca_coords,
        "pca_explained": pca.explained_variance_ratio_,
        "feature_names": list(df_features.columns),
    }


def ensure_cluster_results() -> Tuple[bool, Optional[str]]:
    """Memastikan hasil klaster tersedia & konsisten di session_state berdasarkan data/k/fitur aktif."""
    df_work = st.session_state.get("df_work")
    if df_work is None or df_work.empty:
        return False, "Data kerja tidak tersedia atau kosong. Periksa sumber data dan filter usia."

    selected_features = st.session_state.get("selected_features", DEFAULT_FEATURES)
    feature_cols = get_feature_columns(selected_features)
    k_final = int(st.session_state.get("k_final", DEFAULT_K))

    if len(feature_cols) < 2:
        return False, "Pilih minimal 2 fitur klastering (atur di Halaman 4)."
    if len(df_work) < 3:
        return False, "Data kerja terlalu sedikit untuk klastering."
    if k_final < 2 or k_final >= len(df_work):
        return False, f"Nilai k ({k_final}) tidak valid untuk jumlah data kerja ({len(df_work)}). Atur k di Halaman 4."

    try:
        feat_df = prepare_features(df_work, feature_cols)
        result = run_kmeans(feat_df, k=k_final, random_state=RANDOM_STATE)
        temp = df_work.copy()
        temp["Cluster"] = result["labels"].astype(int)
        cluster_summary, segment_names, gender_comp, prof_comp = build_cluster_profile(temp, "Cluster")
        df_labeled = make_labeled_dataframe(df_work, result["labels"], segment_names)
        seg_color_map = get_segment_color_map(df_labeled["Cluster"], segment_names)
        st.session_state["cluster_result"] = result
        st.session_state["cluster_summary"] = cluster_summary
        st.session_state["segment_names"] = segment_names
        st.session_state["gender_comp"] = gender_comp
        st.session_state["prof_comp"] = prof_comp
        st.session_state["df_labeled"] = df_labeled
        st.session_state["seg_color_map"] = seg_color_map
        return True, None
    except Exception as e:
        return False, f"Gagal menghitung klaster: {e}"


# -----------------------------
# Penamaan segmen & profil
# -----------------------------
def name_segment(
    cluster_income_mean: float,
    cluster_spending_mean: float,
    overall_income: float,
    overall_spending: float,
) -> str:
    """Memberi nama deskriptif segmen berdasarkan posisi Income dan Spending relatif."""

    def level(val: float, ref: float) -> str:
        if ref == 0:
            return "sedang"
        low_th = ref * 0.9
        high_th = ref * 1.1
        if val < low_th:
            return "rendah"
        if val > high_th:
            return "tinggi"
        return "sedang"

    return f"Pendapatan {level(cluster_income_mean, overall_income)} - belanja {level(cluster_spending_mean, overall_spending)}"


def build_cluster_profile(
    df_labeled: pd.DataFrame, label_col: str = "Cluster"
) -> Tuple[pd.DataFrame, Dict[int, str], pd.DataFrame, pd.DataFrame]:
    """Membangun profil klaster: ringkasan, nama segmen, komposisi gender & profesi."""
    overall_income = df_labeled["Annual_Income($)"].mean() if "Annual_Income($)" in df_labeled.columns else 0
    overall_spending = df_labeled["Spending Score(1-100)"].mean() if "Spending Score(1-100)" in df_labeled.columns else 0

    agg_cols = [
        c
        for c in ["Age", "Annual_Income($)", "Spending Score(1-100)", "Work_Experience", "Family_Size"]
        if c in df_labeled.columns
    ]
    summary = df_labeled.groupby(label_col)[agg_cols].mean().reset_index()
    counts = df_labeled.groupby(label_col).size().reset_index(name="Jumlah")
    summary = summary.merge(counts, on=label_col)
    total = len(df_labeled)
    summary["Persentase"] = (summary["Jumlah"] / total * 100).round(2) if total else 0

    segment_names: Dict[int, str] = {}
    for _, row in summary.iterrows():
        cid = int(row[label_col])
        inc = float(row.get("Annual_Income($)", overall_income))
        spd = float(row.get("Spending Score(1-100)", overall_spending))
        segment_names[cid] = name_segment(inc, spd, float(overall_income), float(overall_spending))
    summary["Nama_Segmen"] = summary[label_col].map(segment_names)

    if "Gender" in df_labeled.columns:
        gender_comp = df_labeled.groupby([label_col, "Gender"]).size().reset_index(name="Jumlah")
        gender_tot = gender_comp.groupby(label_col)["Jumlah"].transform("sum")
        gender_comp["Persentase"] = (gender_comp["Jumlah"] / gender_tot * 100).round(2)
        gender_comp["Nama_Segmen"] = gender_comp[label_col].map(segment_names)
    else:
        gender_comp = pd.DataFrame()

    if "Profession" in df_labeled.columns:
        prof_comp = df_labeled.groupby([label_col, "Profession"]).size().reset_index(name="Jumlah")
        prof_tot = prof_comp.groupby(label_col)["Jumlah"].transform("sum")
        prof_comp["Persentase"] = (prof_comp["Jumlah"] / prof_tot * 100).round(2)
        prof_comp["Nama_Segmen"] = prof_comp[label_col].map(segment_names)
    else:
        prof_comp = pd.DataFrame()

    # Profesi dominan per klaster (untuk rekomendasi)
    dominant_prof: Dict[int, str] = {}
    if not prof_comp.empty:
        for cid in summary[label_col].astype(int).tolist():
            sub = prof_comp[prof_comp[label_col] == cid].sort_values("Jumlah", ascending=False)
            if not sub.empty:
                dominant_prof[cid] = str(sub.iloc[0]["Profession"])
    summary["Profesi_Dominan"] = summary[label_col].map(dominant_prof).fillna("-")

    return summary, segment_names, gender_comp, prof_comp


# -----------------------------
# Rekomendasi pemasaran
# -----------------------------
def generate_marketing_recommendations(
    income_mean: float,
    spending_mean: float,
    family_size_mean: float,
    overall_income: float,
    overall_spending: float,
    dominant_profession: Optional[str] = None,
) -> List[str]:
    """Menghasilkan rekomendasi pemasaran berbasis aturan."""
    recs: List[str] = []
    high_income = income_mean > overall_income * 1.1
    low_income = income_mean < overall_income * 0.9
    high_spending = spending_mean > overall_spending * 1.1
    low_spending = spending_mean < overall_spending * 0.9

    # Empat aturan utama (spesifikasi)
    if high_income and high_spending:
        recs.append("Program loyalitas premium (benefit eksklusif, poin/reward, akses prioritas).")
    if high_income and low_spending:
        recs.append("Promosi yang memicu minat belanja (personalisasi, bundling menarik, penawaran terbatas).")
    if low_income and high_spending:
        recs.append("Diskon atau skema cicilan/pembayaran fleksibel untuk menjaga daya beli.")
    if low_income and low_spending:
        recs.append("Promosi hemat berbiaya rendah (value pack, kupon hemat, penawaran dasar).")

    # Tambahan: paket keluarga
    if family_size_mean >= 4:
        recs.append("Paket keluarga/bundle multipack karena rata-rata ukuran keluarga tinggi.")

    # Tambahan: penawaran khusus profesi dominan
    if dominant_profession and dominant_profession != "Unknown":
        recs.append(
            f"Penawaran khusus yang relevan dengan profesi dominan segmen ({dominant_profession})."
        )

    # Fallback jika tidak ada aturan utama yang kena
    if not any(
        x in recs
        for x in [
            "Program loyalitas premium (benefit eksklusif, poin/reward, akses prioritas).",
            "Promosi yang memicu minat belanja (personalisasi, bundling menarik, penawaran terbatas).",
            "Diskon atau skema cicilan/pembayaran fleksibel untuk menjaga daya beli.",
            "Promosi hemat berbiaya rendah (value pack, kupon hemat, penawaran dasar).",
        ]
    ):
        recs.insert(0, "Strategi promosi seimbang dan uji coba penawaran bertahap sesuai perilaku segmen.")

    # Unik, pertahankan urutan
    seen = set()
    uniq = []
    for r in recs:
        if r not in seen:
            uniq.append(r)
            seen.add(r)
    return uniq


# -----------------------------
# Prediksi & keluaran
# -----------------------------
def predict_new_customer(
    model: KMeans, scaler: StandardScaler, input_values: Dict[str, float], feature_cols: List[str]
) -> int:
    """Memprediksi klaster pelanggan baru menggunakan model dan scaler terlatih."""
    row = [input_values[col] for col in feature_cols]
    X = np.array(row, dtype=float).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    return int(pred[0])


def make_labeled_dataframe(df_work: pd.DataFrame, labels: np.ndarray, segment_names: Dict[int, str]) -> pd.DataFrame:
    """Menambahkan kolom Cluster dan Nama_Segmen ke DataFrame kerja."""
    out = df_work.copy()
    out["Cluster"] = labels.astype(int)
    out["Nama_Segmen"] = out["Cluster"].map(segment_names)
    return out


# -----------------------------
# Ringkasan & jawaban otomatis
# -----------------------------
def build_auto_summary(
    cluster_summary: pd.DataFrame, k: int, silhouette: float, overall_means: pd.Series
) -> Dict[str, Any]:
    """Membangun ringkasan temuan otomatis dari hasil klaster."""
    sizes = []
    for _, row in cluster_summary.sort_values("Cluster").iterrows():
        sizes.append(
            {
                "cluster": int(row["Cluster"]),
                "nama": row.get("Nama_Segmen", ""),
                "jumlah": int(row.get("Jumlah", 0)),
                "persentase": float(row.get("Persentase", 0)),
                "income": float(row.get("Annual_Income($)", 0)),
                "spending": float(row.get("Spending Score(1-100)", 0)),
            }
        )
    largest = max(sizes, key=lambda x: x["jumlah"]) if sizes else None
    smallest = min(sizes, key=lambda x: x["jumlah"]) if sizes else None
    return {
        "k": k,
        "silhouette": silhouette,
        "sizes": sizes,
        "largest": largest,
        "smallest": smallest,
        "overall_means": overall_means.to_dict() if hasattr(overall_means, "to_dict") else {},
    }


def build_research_answers(
    k: int,
    selected_features_display: List[str],
    silhouette: float,
    cluster_summary: pd.DataFrame,
    segment_names: Dict[int, str],
) -> Dict[str, str]:
    """Menyusun jawaban singkat untuk tiap rumusan masalah berdasarkan hasil aktual."""
    sil_text = f"{silhouette:.4f}" if silhouette == silhouette else "N/A"
    seg_lines = []
    for _, row in cluster_summary.sort_values("Cluster").iterrows():
        seg_lines.append(
            f"Klaster {int(row['Cluster'])} ({row.get('Nama_Segmen','')}, n={int(row['Jumlah'])}, {row.get('Persentase',0)}%)"
        )
    seg_str = "; ".join(seg_lines) if seg_lines else "-"

    rm1 = (
        f"Pelanggan dikelompokkan menjadi {k} klaster menggunakan K-Means pada fitur terstandarisasi: "
        f"{', '.join(selected_features_display)}. Hasil menunjukkan {seg_str}."
    )
    rm2 = (
        f"Jumlah klaster dipilih k={k} berdasarkan pertimbangan Elbow dan Silhouette Score (k=2–10) dengan random_state tetap. "
        f"Silhouette Score k terpilih = {sil_text}, yang mengindikasikan pemisahan relatif lemah sehingga klaster bersifat pengelompokan praktis."
    )
    profiles = []
    for _, row in cluster_summary.sort_values("Cluster").iterrows():
        profiles.append(
            f"K{int(row['Cluster'])}:{row.get('Nama_Segmen','')} "
            f"(Income≈{row.get('Annual_Income($)',0):.0f}, Spending≈{row.get('Spending Score(1-100)',0):.1f}, "
            f"Gender/Profesi dilihat dari komposisi dominan)"
        )
    rm3 = "Profil tiap segmen diringkas dari rata-rata atribut dan komposisi gender/profesi: " + " | ".join(profiles)
    rm4 = (
        "Strategi pemasaran dirumuskan berbasis aturan: loyalitas premium (pendapatan tinggi-belanja tinggi), "
        "promosi pemicu belanja (pendapatan tinggi-belanja rendah), diskon/cicilan (pendapatan rendah-belanja tinggi), "
        "promosi hemat (pendapatan rendah-belanja rendah), ditambah paket keluarga bila Family_Size tinggi dan penawaran sesuai profesi dominan."
    )
    return {"RM1": rm1, "RM2": rm2, "RM3": rm3, "RM4": rm4}


def rename_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Mengganti nama kolom ke label tampilan yang rapi."""
    return df.rename(columns=DISPLAY_LABELS)


def get_column_role_df() -> pd.DataFrame:
    """Mengembalikan tabel peran kolom untuk Halaman 2."""
    rows = []
    for col in REQUIRED_COLUMNS:
        if col in CLUSTER_FORMING_COLS:
            role = "Pembentuk klaster (variabel numerik, dapat dipilih & distandardisasi)"
        else:
            role = "Pelengkap profil/deskriptif (tidak wajib masuk model klaster)"
        rows.append({"Kolom": col, "Label": DISPLAY_LABELS.get(col, col), "Peran": role})
    return pd.DataFrame(rows)
