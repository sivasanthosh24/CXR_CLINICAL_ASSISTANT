"""Shared, restrained CXR Clinical AI visual theme."""

import streamlit as st


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --cxr-ink: #152238;
            --cxr-muted: #62728a;
            --cxr-accent: #147d86;
            --cxr-accent-soft: #e8f3f3;
            --cxr-page: #f3f6fa;
            --cxr-line: #dce4ed;
            --cxr-card: #ffffff;
            --cxr-sidebar: #142238;
        }
        .stApp { background: var(--cxr-page); color: var(--cxr-ink); }
        [data-testid="stAppViewContainer"] { background: var(--cxr-page); }
        .block-container {
            max-width: 1440px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        header[data-testid="stHeader"] { background: rgba(243, 246, 250, .94); }
        [data-testid="stSidebar"] { background: var(--cxr-sidebar); }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span { color: #e9eff7; }
        [data-testid="stSidebarNav"] a {
            border-radius: 8px;
            margin: 2px 8px;
            transition: background .15s ease;
        }
        [data-testid="stSidebarNav"] a:hover { background: #243752; }
        [data-testid="stSidebarNav"] a[aria-current="page"] {
            background: #2c4564;
            font-weight: 650;
        }
        [data-testid="stSidebarNav"] a[aria-current="page"] span { color: #ffffff; }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
        h1, h2, h3 { color: var(--cxr-ink); letter-spacing: -.018em; }
        [data-testid="stMetric"] {
            background: var(--cxr-card);
            border: 1px solid var(--cxr-line);
            border-radius: 12px;
            padding: 14px 16px;
            box-shadow: 0 2px 8px rgba(21, 34, 56, .035);
            min-height: 106px;
        }
        [data-testid="stMetricLabel"] { color: var(--cxr-muted); }
        [data-testid="stMetricValue"] { color: var(--cxr-ink); font-weight: 700; }
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            border: 1px solid var(--cxr-line);
            border-radius: 10px;
            overflow: hidden;
            background: var(--cxr-card);
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--cxr-line);
            border-radius: 14px;
            background: rgba(255, 255, 255, .76);
        }
        .stAlert { border-radius: 10px; }
        .stButton > button {
            border-radius: 9px;
            font-weight: 600;
            transition: all .15s ease;
        }
        .stButton > button[kind="primary"] {
            background: #176f78;
            border-color: #176f78;
        }
        .stButton > button[kind="primary"]:hover {
            background: #125d65;
            border-color: #125d65;
        }
        [data-testid="stFileUploaderDropzone"] {
            border-color: #8299ad;
            border-radius: 10px;
            background: rgba(255, 255, 255, .06);
        }
        .page-hero {
            background: linear-gradient(110deg, #d2ece8 0%, #e1f2ef 58%, #edf5f6 100%);
            border: 1px solid #afd4ce;
            border-left: 5px solid var(--cxr-accent);
            border-radius: 14px;
            padding: 22px 26px 20px;
            margin: 0 0 24px;
            box-shadow: 0 5px 18px rgba(20, 125, 134, .13);
        }
        .page-eyebrow {
            color: var(--cxr-accent);
            font-size: 11px;
            font-weight: 750;
            letter-spacing: .11em;
            text-transform: uppercase;
            margin-bottom: 5px;
        }
        .page-title {
            color: var(--cxr-ink);
            font-size: 30px;
            line-height: 1.2;
            font-weight: 760;
            margin: 0;
        }
        .page-subtitle {
            color: var(--cxr-muted);
            font-size: 14px;
            line-height: 1.55;
            margin: 7px 0 0;
        }
        .section-title {
            color: var(--cxr-ink);
            font-size: 18px;
            font-weight: 720;
            background: var(--cxr-accent-soft);
            border-left: 4px solid var(--cxr-accent);
            border-radius: 7px;
            padding: 9px 13px;
            margin: 22px 0 12px;
        }
        .section-header {
            color: #ffffff;
            background: var(--cxr-sidebar);
            border-left: 4px solid #3ea4a4;
            border-radius: 9px;
            padding: 12px 16px;
            margin: 24px 0 13px;
            font-size: 17px;
            font-weight: 700;
        }
        .info-card, .status-card {
            background: var(--cxr-card);
            border: 1px solid var(--cxr-line);
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(21, 34, 56, .035);
        }
        .note {
            background: #f8fbfc;
            border-left-color: var(--cxr-accent);
            border-radius: 8px;
            color: #46576d;
        }
        .footer {
            color: var(--cxr-muted);
            border-top: 1px solid var(--cxr-line);
        }
        @media (max-width: 800px) {
            .block-container { padding: 1rem 1rem 2rem; }
            .page-hero { padding: 18px; }
            .page-title { font-size: 25px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def display_table(dataframe, **kwargs) -> None:
    """Render a Streamlit table with the shared header and row styling."""
    styled = dataframe.style.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", "#e8f0f4"),
                    ("color", "#20334b"),
                    ("font-weight", "700"),
                    ("border-bottom", "1px solid #cbd8e2"),
                    ("padding", "10px 12px"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("color", "#24364e"),
                    ("border-bottom", "1px solid #e7edf2"),
                    ("padding", "9px 12px"),
                ],
            },
            {
                "selector": "tbody tr:nth-child(even)",
                "props": [("background-color", "#f7fafc")],
            },
        ],
        overwrite=False,
    )
    st.dataframe(styled, **kwargs)
