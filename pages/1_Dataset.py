from __future__ import annotations

import streamlit as st
import pandas as pd
from src.ui.theme import display_table, inject_theme


st.set_page_config(
    page_title="Dataset | CXR Clinical AI",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .stApp {
        background: #f6f8fb;
    }

    .block-container {
        max-width: 1320px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .page-title {
        font-size: 30px;
        font-weight: 750;
        color: #0f172a;
        margin-bottom: 2px;
    }

    .page-subtitle {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 22px;
    }

    .section-title {
        color: #0f172a;
        font-size: 20px;
        font-weight: 700;
        margin-top: 18px;
        margin-bottom: 10px;
    }

    .info-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
        min-height: 105px;
    }

    .info-label {
        color: #64748b;
        font-size: 13px;
        margin-bottom: 5px;
    }

    .info-value {
        color: #0f172a;
        font-size: 25px;
        font-weight: 750;
    }

    .info-note {
        color: #64748b;
        font-size: 12px;
        margin-top: 3px;
    }

    .status-card {
        background: white;
        border: 1px solid #dbeafe;
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 10px;
    }

    .status-title {
        color: #0f172a;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .status-text {
        color: #475569;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
inject_theme()


# ================================================================
# VERIFIED STATIC DATA
# ================================================================

DISEASES = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]


SPLIT_DATA = pd.DataFrame(
    {
        "Split": ["Train", "Validation", "Test"],
        "Images": [7000, 1500, 1500],
        "Percentage": [70.00, 15.00, 15.00],
    }
)


VIEW_DATA = pd.DataFrame(
    {
        "View": ["AP", "Lateral"],
        "Images": [6384, 3616],
        "Percentage": [63.84, 36.16],
    }
)


VIEW_SPLIT_DATA = pd.DataFrame(
    {
        "Split": ["Train", "Validation", "Test"],
        "AP": [4447, 958, 979],
        "Lateral": [2553, 542, 521],
    }
)


VIEW_SPLIT_PERCENTAGE_DATA = pd.DataFrame(
    {
        "Split": ["Train", "Validation", "Test"],
        "AP (%)": [63.53, 63.87, 65.27],
        "Lateral (%)": [36.47, 36.13, 34.73],
    }
)


SUBJECT_SPLIT_DATA = pd.DataFrame(
    {
        "Split": ["Train", "Validation", "Test"],
        "Subjects": [5181, 1125, 1098],
    }
)


DISEASE_LABEL_DATA = pd.DataFrame(
    {
        "Disease": DISEASES,
        "Positive": [
            536, 2641, 2632, 342, 1387, 519, 466,
            2536, 312, 2484, 129, 355, 3544, 2105
        ],
        "Negative": [
            2965, 2192, 199, 35, 1783, 2463, 1077,
            29, 5852, 4534, 0, 231, 39, 7895
        ],
        "Uncertain": [
            1230, 540, 5, 50, 610, 294, 1013,
            513, 70, 337, 51, 18, 2, 0
        ],
        "Blank": [
            5269, 4627, 7164, 9573, 6220, 6724, 7444,
            6922, 3766, 2645, 9820, 9396, 6415, 0
        ],
    }
)


VALIDATION_DATA = pd.DataFrame(
    {
        "Validation Check": [
            "Master dataset rows",
            "Unique image IDs",
            "Duplicate image IDs",
            "Duplicate subject-image pairs",
            "Images found",
            "Images missing",
            "Subjects appearing in multiple splits",
            "Patient-level split leakage",
        ],
        "Result": [
            "10,000",
            "10,000",
            "0",
            "0",
            "10,000",
            "0",
            "0",
            "NONE",
        ],
    }
)


# ================================================================
# PAGE HEADER
# ================================================================

st.markdown(
    """
    <div class="page-hero">
      <div class="page-eyebrow">CXR Clinical AI · Reference</div>
      <div class="page-title">Dataset Overview</div>
      <div class="page-subtitle">Dataset composition, image views, disease labels,
      and patient-level validation for the CXR Clinical AI system.</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ================================================================
# DATASET SUMMARY
# ================================================================

st.markdown(
    '<div class="section-title">Dataset Summary</div>',
    unsafe_allow_html=True,
)

summary_cols = st.columns(5)

summary_items = [
    ("Total Images", "10,000", "Rows in the master dataset"),
    ("Subjects", "7,404", "Unique subject IDs"),
    ("Studies", "9,853", "Unique study IDs"),
    ("Columns", "71", "Master dataset columns"),
    ("Disease Classes", "14", "Target disease categories"),
]

for column, (label, value, note) in zip(summary_cols, summary_items):
    with column:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-label">{label}</div>
                <div class="info-value">{value}</div>
                <div class="info-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ================================================================
# DATASET SPLIT + OVERALL VIEW DISTRIBUTION
# ================================================================

st.markdown(
    '<div class="section-title">Dataset Composition</div>',
    unsafe_allow_html=True,
)

left, right = st.columns(2)

with left:
    st.markdown("**Dataset Split**")
    display_table(
        SPLIT_DATA,
        column_config={
            "Percentage": st.column_config.NumberColumn(
                "Percentage", format="%.2f%%"
            )
        },
        hide_index=True,
        width="stretch",
    )

with right:
    st.markdown("**Overall View Distribution**")
    display_table(
        VIEW_DATA,
        column_config={
            "Percentage": st.column_config.NumberColumn(
                "Percentage", format="%.2f%%"
            )
        },
        hide_index=True,
        width="stretch",
    )


# ================================================================
# VIEW DISTRIBUTION BY SPLIT
# ================================================================

st.markdown(
    '<div class="section-title">View Distribution by Split</div>',
    unsafe_allow_html=True,
)

display_table(
    VIEW_SPLIT_DATA,
    hide_index=True,
    width="stretch",
)

st.markdown("**View Percentages Within Each Split**")

display_table(
    VIEW_SPLIT_PERCENTAGE_DATA,
    column_config={
        "AP (%)": st.column_config.NumberColumn("AP (%)", format="%.2f%%"),
        "Lateral (%)": st.column_config.NumberColumn(
            "Lateral (%)", format="%.2f%%"
        ),
    },
    hide_index=True,
    width="stretch",
)


# ================================================================
# PATIENT DISTRIBUTION BY SPLIT
# ================================================================

st.markdown(
    '<div class="section-title">Patient Distribution by Split</div>',
    unsafe_allow_html=True,
)

display_table(
    SUBJECT_SPLIT_DATA,
    hide_index=True,
    width="stretch",
)


# ================================================================
# PATIENT-LEVEL SPLIT VALIDATION
# ================================================================

st.markdown(
    '<div class="section-title">Patient-Level Split Validation</div>',
    unsafe_allow_html=True,
)

left, right = st.columns(2)

with left:
    st.markdown(
        """
        <div class="status-card">
            <div class="status-title">
                ✓ Patient-level leakage: NONE
            </div>
            <div class="status-text">
                No subjects appeared in more than one dataset split.
            </div>
        </div>

        <div class="status-card">
            <div class="status-title">
                ✓ Unique image IDs
            </div>
            <div class="status-text">
                10,000 unique image IDs; duplicate image IDs: 0.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        """
        <div class="status-card">
            <div class="status-title">
                ✓ Image availability
            </div>
            <div class="status-text">
                10,000 of 10,000 expected images were found
                (100.00%).
            </div>
        </div>

        <div class="status-card">
            <div class="status-title">
                ✓ Duplicate subject-image pairs
            </div>
            <div class="status-text">
                No duplicate subject-image pairs were identified.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ================================================================
# IMAGE PROPERTIES
# ================================================================

st.markdown(
    '<div class="section-title">Image Properties</div>',
    unsafe_allow_html=True,
)

image_cols = st.columns(4)

image_properties = [
    ("Format", "JPEG"),
    ("Image Mode", "RGB"),
    ("Width", "512 px"),
    ("Height", "512 px"),
]

for column, (label, value) in zip(image_cols, image_properties):
    with column:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-label">{label}</div>
                <div class="info-value">{value}</div>
                <div class="info-note">Sample inspection</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ================================================================
# FULL DISEASE LABEL DISTRIBUTION
# ================================================================

st.markdown(
    '<div class="section-title">Disease Label Distribution</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Raw disease-label distribution before conversion into the "
    "supervised target and mask representation."
)

display_table(
    DISEASE_LABEL_DATA,
    hide_index=True,
    width="stretch",
)


# ================================================================
# DATASET STRUCTURE
# ================================================================

st.markdown(
    '<div class="section-title">Dataset Structure</div>',
    unsafe_allow_html=True,
)

st.markdown(
    "The dataset contains image-level records with subject and "
    "study identifiers, image paths, normalized view information, "
    "split assignments, report-related fields, disease labels, "
    "and target/mask fields for the 14 disease classes. The final "
    "master dataset contains 10,000 image records across training, "
    "validation, and test partitions."
)


# ================================================================
# DATASET VALIDATION SUMMARY
# ================================================================

st.markdown(
    '<div class="section-title">Dataset Validation Summary</div>',
    unsafe_allow_html=True,
)

display_table(
    VALIDATION_DATA,
    hide_index=True,
    width="stretch",
)


# ================================================================
# FOOTER
# ================================================================

st.caption(
    "Dataset statistic and distribution"
)
