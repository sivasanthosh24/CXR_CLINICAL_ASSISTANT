from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
import pandas as pd
from src.ui.theme import display_table, inject_theme


# ================================================================
# PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="CXR Clinical AI — Evaluation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ================================================================
# STATIC VERIFIED EVALUATION RESULTS
# ================================================================
# This page uses saved, verified evaluation values.
#
# It does NOT:
#   - load model checkpoints
#   - run inference
#   - calculate metrics
#   - retrain models
#   - access the image dataset
#
# Values below are the final evaluation results obtained from the
# project's verified evaluation runs / source documents. Module 3 reads
# only the metadata in its saved index for coverage and input-quality counts.
# ================================================================


# ----------------------------------------------------------------
# MODULE 1.1 — VIEW CLASSIFICATION
# ----------------------------------------------------------------

VIEW_ARCHITECTURE = {
    "Architecture": "DenseNet121 → Global Average Pooling → FC(2) → Softmax",
    "Classes": "0 = AP, 1 = Lateral",
    "PA used": "No",
}

VIEW_METRICS = {
    "Accuracy": 0.9993,
    "Precision": 0.9993,
    "Recall": 0.9993,
    "F1 Score": 0.9993,
}


# ----------------------------------------------------------------
# MODULE 1 — DISEASE PREDICTION
# ----------------------------------------------------------------
# Final Module 1 disease-model test evaluation reported in the
# project evaluation document.
# ----------------------------------------------------------------

DISEASE_METRICS = {
    "Mean Test AUROC": 0.78919,
    "Mean Test Average Precision": 0.75152,
    "Mean Test F1": 0.75001,
    "Mean Test Accuracy": 0.84418,
    "Mean Test Precision": 0.70117,
    "Mean Test Recall": 0.82407,
    "Mean Test Specificity": 0.48273,
}

DISEASE_TEST_INFO = {
    "Test images": 1500,
    "AP images": 979,
    "Lateral images": 521,
}


# ----------------------------------------------------------------
# MODULE 2 V2 — CLINICAL REPORT GENERATION
# ----------------------------------------------------------------

MODULE2_ARCHITECTURE = {
    "Vision encoder": "BiomedCLIP ViT",
    "Vision model": "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "Vision transformer blocks": 12,
    "Frozen ViT blocks": "0–8",
    "Trainable ViT blocks": "9–11",
    "Visual token sequence": "197 × 768",
    "Decoder": "GPT-2 with cross-attention",
    "GPT-2 layers": 12,
    "GPT-2 attention heads": 12,
    "LoRA rank": 8,
    "LoRA alpha": 16,
    "LoRA dropout": 0.10,
    "Cross-attention": "Enabled",
    "Total trainable parameters": 51_751_680,
}

MODULE2_TRAINING = {
    "Learning rate": "1 × 10⁻⁴",
    "Weight decay": 0.01,
    "Warmup ratio": 0.10,
    "Optimizer": "AdamW",
    "Scheduler": "Linear warmup + cosine decay",
    "Early stopping patience": 2,
}

MODULE2_TEST_METRICS = {
    "Test samples": 1439,
    "Non-empty reports": 1439,
    "Empty reports": 0,
    "Empty report rate": 0.0,
    "Reference mean words": 59.756081,
    "Reference median words": 54.0,
    "Generated mean words": 63.140375,
    "Generated median words": 63.0,
    "Reference max words": 193,
    "Generated max words": 79,
    "Test loss": 1.542568,
    "Test perplexity": 4.676583,
}

MODULE2_GENERATION_METRICS = {
    "BLEU-1": 0.244710,
    "BLEU-2": 0.136886,
    "BLEU-3": 0.086596,
    "BLEU-4": 0.058915,

    "ROUGE-1 Precision": 0.329402,
    "ROUGE-1 Recall": 0.396822,
    "ROUGE-1 F1": 0.338539,

    "ROUGE-2 Precision": 0.115137,
    "ROUGE-2 Recall": 0.151157,
    "ROUGE-2 F1": 0.122528,

    "ROUGE-L Precision": 0.221084,
    "ROUGE-L Recall": 0.275913,
    "ROUGE-L F1": 0.229866,

    "BERTScore Precision": 0.832500,
    "BERTScore Recall": 0.830970,
    "BERTScore F1": 0.831180,

    "TF-IDF Cosine Mean": 0.095690,
    "TF-IDF Cosine Median": 0.082292,
    "TF-IDF Cosine Std": 0.057766,
}

MODULE2_GENERATION_QUALITY = {
    "Mean trigram repetition rate": 0.003197,
    "Median trigram repetition rate": 0.0,
    "Reports with repetition": 253,
    "Reports with high repetition": 0,
    "Mean generated tokens": 96.0,
    "Generation throughput (samples/sec)": 2.065748,
    "Generation time (minutes)": 11.610000,
    "BERTScore time (seconds)": 14.499020,
}


# ----------------------------------------------------------------
# MODULE 3 — SAVED INDEX COVERAGE / DATA QUALITY
# ----------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE3_INDEX_PATH = PROJECT_ROOT / ".runtime" / "module3" / "image_index.npz"


def load_module3_index_evaluation():
    """Read only the small metadata member; never load the embedding matrix."""
    if not MODULE3_INDEX_PATH.is_file():
        return None
    try:
        with np.load(MODULE3_INDEX_PATH, allow_pickle=False) as index_file:
            metadata = json.loads(str(index_file["metadata"].item()))
        stats = metadata["stats"]
        total = sum(
            int(stats.get(key, 0))
            for key in ("indexed", "missing_images", "invalid_images", "invalid_text_or_id")
        )
        return {
            "indexed": int(stats.get("indexed", 0)),
            "missing_images": int(stats.get("missing_images", 0)),
            "invalid_images": int(stats.get("invalid_images", 0)),
            "invalid_text_or_id": int(stats.get("invalid_text_or_id", 0)),
            "total": total,
            "coverage": (int(stats.get("indexed", 0)) / total) if total else 0.0,
            "model": f"{metadata.get('model', 'Unknown')} · {metadata.get('pretrained', 'Unknown')}",
            "size_mb": MODULE3_INDEX_PATH.stat().st_size / (1024 * 1024),
        }
    except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError):
        return None


MODULE3_INDEX_EVALUATION = load_module3_index_evaluation()


# ================================================================
# STYLING
# ================================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #f6f8fb;
    }

    .block-container {
        max-width: 1320px;
        padding-top: 4.5rem !important;
        padding-bottom: 2.5rem;
    }

    /* Keep the page title clear of Streamlit's top toolbar/header. */
    header[data-testid="stHeader"] {
        z-index: 1000;
    }

    .page-title {
        position: relative;
        z-index: 1;
        line-height: 1.25;
    }

    .page-title {
        font-size: 30px;
        font-weight: 750;
        color: #0f172a;
        margin-bottom: 3px;
    }

    .page-subtitle {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 22px;
    }

    .section-header {
        background: #0f172a;
        color: white;
        border-radius: 10px;
        padding: 13px 17px;
        margin-top: 18px;
        margin-bottom: 14px;
        font-size: 18px;
        font-weight: 700;
    }

    .info-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 15px 17px;
        margin-bottom: 12px;
    }

    .metric-label {
        color: #64748b;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    .metric-value {
        color: #0f172a;
        font-size: 25px;
        font-weight: 750;
        margin-top: 3px;
    }

    .note {
        color: #475569;
        background: #f8fafc;
        border-left: 4px solid #94a3b8;
        padding: 10px 13px;
        border-radius: 5px;
        font-size: 13px;
        margin: 10px 0 14px 0;
    }

    .footer {
        color: #64748b;
        font-size: 12px;
        text-align: center;
        margin-top: 28px;
        padding-top: 14px;
        border-top: 1px solid #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
inject_theme()


# ================================================================
# HEADER
# ================================================================

st.markdown(
    """
    <div class="page-hero">
      <div class="page-eyebrow">CXR Clinical AI · Model review</div>
      <div class="page-title">Evaluation & Performance</div>
      <div class="page-subtitle">Verified evaluation results for view classification,
      disease prediction, clinical report generation, and Module 3 retrieval index coverage.</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ================================================================
# EVALUATION OVERVIEW
# ================================================================

st.markdown(
    '<div class="section-header">Evaluation Overview</div>',
    unsafe_allow_html=True,
)

overview_cols = st.columns(4)

with overview_cols[0]:
    st.markdown(
        """
        <div class="info-card">
        <b>Module 1.1</b><br>
        AP / Lateral View Classification<br>
        <span style="color:#64748b;font-size:13px;">
        DenseNet121-based binary classification
        </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with overview_cols[1]:
    st.markdown(
        """
        <div class="info-card">
        <b>Module 1</b><br>
        Multi-disease Prediction<br>
        <span style="color:#64748b;font-size:13px;">
        14-disease chest X-ray prediction
        </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with overview_cols[2]:
    st.markdown(
        """
        <div class="info-card">
        <b>Module 2</b><br>
        Clinical Report Generation<br>
        <span style="color:#64748b;font-size:13px;">
        BiomedCLIP + GPT-2 with visual cross-attention
        </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with overview_cols[3]:
    st.markdown(
        """
        <div class="info-card">
        <b>Module 3</b><br>
        Similar Image Retrieval<br>
        <span style="color:#64748b;font-size:13px;">
        CPU visual embeddings and cosine ranking
        </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ================================================================
# MODULE 1.1 — VIEW CLASSIFICATION
# ================================================================

st.markdown(
    '<div class="section-header">1. Module 1.1 — View Classification Evaluation</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="note">'
    "<b>Task:</b> classify a chest X-ray into AP or Lateral view. "
    "The reported final test result was obtained using the trained "
    "DenseNet121 classifier."
    "</div>",
    unsafe_allow_html=True,
)

architecture_df = pd.DataFrame(
    list(VIEW_ARCHITECTURE.items()),
    columns=["Configuration", "Value"],
)

display_table(
    architecture_df,
    width="stretch",
    hide_index=True,
)

view_cols = st.columns(4)

for col, (metric_name, metric_value) in zip(
    view_cols,
    VIEW_METRICS.items(),
):
    with col:
        st.metric(
            metric_name,
            f"{metric_value:.4f}",
        )

st.markdown("**Final Test Performance**")

view_result_df = pd.DataFrame(
    {
        "Metric": list(VIEW_METRICS.keys()),
        "Test Value": [
            f"{value:.4f}"
            for value in VIEW_METRICS.values()
        ],
    }
)

display_table(
    view_result_df,
    width="stretch",
    hide_index=True,
)


# ================================================================
# MODULE 1 — DISEASE PREDICTION
# ================================================================

st.markdown(
    '<div class="section-header">2. Module 1 — Disease Prediction Evaluation</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="note">'
    "<b>Evaluation:</b> final Module 1 disease-model performance on the "
    "original 1,500-image test set. The reported values are mean test "
    "metrics across the 14 disease prediction outputs."
    "</div>",
    unsafe_allow_html=True,
)

disease_info_cols = st.columns(3)

with disease_info_cols[0]:
    st.metric("Test Images", f"{DISEASE_TEST_INFO['Test images']:,}")

with disease_info_cols[1]:
    st.metric("AP Images", f"{DISEASE_TEST_INFO['AP images']:,}")

with disease_info_cols[2]:
    st.metric("Lateral Images", f"{DISEASE_TEST_INFO['Lateral images']:,}")

st.markdown("**Final Test Metrics**")

disease_metric_cols = st.columns(4)

disease_metric_items = list(DISEASE_METRICS.items())

for col, (metric_name, metric_value) in zip(
    disease_metric_cols,
    disease_metric_items[:4],
):
    with col:
        st.metric(
            metric_name.replace("Mean Test ", ""),
            f"{metric_value:.5f}",
        )

disease_metric_cols_2 = st.columns(3)

for col, (metric_name, metric_value) in zip(
    disease_metric_cols_2,
    disease_metric_items[4:],
):
    with col:
        st.metric(
            metric_name.replace("Mean Test ", ""),
            f"{metric_value:.5f}",
        )

disease_result_df = pd.DataFrame(
    {
        "Metric": list(DISEASE_METRICS.keys()),
        "Test Result": [
            f"{value:.5f}"
            for value in DISEASE_METRICS.values()
        ],
    }
)

display_table(
    disease_result_df,
    width="stretch",
    hide_index=True,
)


# ================================================================
# MODULE 2 — ARCHITECTURE
# ================================================================

st.markdown(
    '<div class="section-header">3. Module 2 — Clinical Narrative Generation</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="note">'
    "<b>Task:</b> generate a clinical chest X-ray report from the image. "
    "The final verified V2 architecture uses BiomedCLIP visual features "
    "as conditioning information for a GPT-2 decoder with cross-attention "
    "and LoRA fine-tuning."
    "</div>",
    unsafe_allow_html=True,
)

architecture_module2_df = pd.DataFrame(
    [(name, str(value)) for name, value in MODULE2_ARCHITECTURE.items()],
    columns=["Component", "Configuration"],
)

display_table(
    architecture_module2_df,
    width="stretch",
    hide_index=True,
)


# ================================================================
# MODULE 2 — TRAINING CONFIGURATION
# ================================================================

st.markdown("**Training Configuration**")

training_df = pd.DataFrame(
    [(name, str(value)) for name, value in MODULE2_TRAINING.items()],
    columns=["Parameter", "Value"],
)

display_table(
    training_df,
    width="stretch",
    hide_index=True,
)


# ================================================================
# MODULE 2 — TEST LOSS / PERPLEXITY
# ================================================================

st.markdown(
    '<div class="section-header">4. Module 2 — Clinical Narrative Generation: Test Evaluation</div>',
    unsafe_allow_html=True,
)

module2_test_cols = st.columns(4)

test_metric_display = [
    ("Test Loss", MODULE2_TEST_METRICS["Test loss"], ".6f"),
    ("Test Perplexity", MODULE2_TEST_METRICS["Test perplexity"], ".6f"),
    ("Test Samples", MODULE2_TEST_METRICS["Test samples"], ","),
    ("Non-empty Reports", MODULE2_TEST_METRICS["Non-empty reports"], ","),
]

for col, (label, value, fmt) in zip(
    module2_test_cols,
    test_metric_display,
):
    with col:
        if fmt == ",":
            display_value = f"{int(value):,}"
        else:
            display_value = format(value, fmt)

        st.metric(label, display_value)

module2_basic_df = pd.DataFrame(
    {
        "Generation Statistic": [
            "Empty reports",
            "Empty report rate",
            "Reference mean words",
            "Reference median words",
            "Generated mean words",
            "Generated median words",
            "Reference maximum words",
            "Generated maximum words",
        ],
        "Result": [
            f"{MODULE2_TEST_METRICS['Empty reports']:,}",
            f"{MODULE2_TEST_METRICS['Empty report rate']:.1%}",
            f"{MODULE2_TEST_METRICS['Reference mean words']:.6f}",
            f"{MODULE2_TEST_METRICS['Reference median words']:.1f}",
            f"{MODULE2_TEST_METRICS['Generated mean words']:.6f}",
            f"{MODULE2_TEST_METRICS['Generated median words']:.1f}",
            f"{MODULE2_TEST_METRICS['Reference max words']:,}",
            f"{MODULE2_TEST_METRICS['Generated max words']:,}",
        ],
    }
)

display_table(
    module2_basic_df,
    width="stretch",
    hide_index=True,
)


# ================================================================
# MODULE 2 — GENERATION METRICS
# ================================================================

st.markdown(
    '<div class="section-header">5. Module 2 — Clinical Narrative Generation Metrics</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="note">'
    "The generation evaluation compares the generated clinical reports "
    "with the corresponding reference reports using BLEU, ROUGE, "
    "BERTScore, TF-IDF cosine similarity, and repetition analysis."
    "</div>",
    unsafe_allow_html=True,
)

# BLEU
st.markdown("**BLEU Scores**")

bleu_df = pd.DataFrame(
    {
        "Metric": [
            "BLEU-1",
            "BLEU-2",
            "BLEU-3",
            "BLEU-4",
        ],
        "Score": [
            MODULE2_GENERATION_METRICS["BLEU-1"],
            MODULE2_GENERATION_METRICS["BLEU-2"],
            MODULE2_GENERATION_METRICS["BLEU-3"],
            MODULE2_GENERATION_METRICS["BLEU-4"],
        ],
    }
)

bleu_df["Score"] = bleu_df["Score"].map(lambda x: f"{x:.6f}")

display_table(
    bleu_df,
    width="stretch",
    hide_index=True,
)

# ROUGE
st.markdown("**ROUGE F1 Scores**")

rouge_df = pd.DataFrame(
    {
        "Metric": [
            "ROUGE-1",
            "ROUGE-2",
            "ROUGE-L",
        ],
        "Precision": [
            MODULE2_GENERATION_METRICS["ROUGE-1 Precision"],
            MODULE2_GENERATION_METRICS["ROUGE-2 Precision"],
            MODULE2_GENERATION_METRICS["ROUGE-L Precision"],
        ],
        "Recall": [
            MODULE2_GENERATION_METRICS["ROUGE-1 Recall"],
            MODULE2_GENERATION_METRICS["ROUGE-2 Recall"],
            MODULE2_GENERATION_METRICS["ROUGE-L Recall"],
        ],
        "F1": [
            MODULE2_GENERATION_METRICS["ROUGE-1 F1"],
            MODULE2_GENERATION_METRICS["ROUGE-2 F1"],
            MODULE2_GENERATION_METRICS["ROUGE-L F1"],
        ],
    }
)

for column in ["Precision", "Recall", "F1"]:
    rouge_df[column] = rouge_df[column].map(lambda x: f"{x:.6f}")

display_table(
    rouge_df,
    width="stretch",
    hide_index=True,
)

# BERTScore + cosine
st.markdown("**Semantic Similarity Metrics**")

semantic_df = pd.DataFrame(
    {
        "Metric": [
            "BERTScore Precision",
            "BERTScore Recall",
            "BERTScore F1",
            "TF-IDF Cosine Mean",
            "TF-IDF Cosine Median",
            "TF-IDF Cosine Std",
        ],
        "Score": [
            MODULE2_GENERATION_METRICS["BERTScore Precision"],
            MODULE2_GENERATION_METRICS["BERTScore Recall"],
            MODULE2_GENERATION_METRICS["BERTScore F1"],
            MODULE2_GENERATION_METRICS["TF-IDF Cosine Mean"],
            MODULE2_GENERATION_METRICS["TF-IDF Cosine Median"],
            MODULE2_GENERATION_METRICS["TF-IDF Cosine Std"],
        ],
    }
)

semantic_df["Score"] = semantic_df["Score"].map(lambda x: f"{x:.6f}")

display_table(
    semantic_df,
    width="stretch",
    hide_index=True,
)


# ================================================================
# MODULE 2 — GENERATION QUALITY / REPETITION
# ================================================================

st.markdown(
    '<div class="section-header">6. Module 2 — Clinical Narrative Generation Statistics</div>',
    unsafe_allow_html=True,
)

quality_cols = st.columns(4)

quality_items = [
    (
        "Mean Trigram Repetition",
        MODULE2_GENERATION_QUALITY["Mean trigram repetition rate"],
        ".6f",
    ),
    (
        "Median Trigram Repetition",
        MODULE2_GENERATION_QUALITY["Median trigram repetition rate"],
        ".6f",
    ),
    (
        "Reports With Repetition",
        MODULE2_GENERATION_QUALITY["Reports with repetition"],
        ",",
    ),
    (
        "High Repetition Reports",
        MODULE2_GENERATION_QUALITY["Reports with high repetition"],
        ",",
    ),
]

for col, (label, value, fmt) in zip(
    quality_cols,
    quality_items,
):
    with col:
        if fmt == ",":
            display_value = f"{int(value):,}"
        else:
            display_value = format(value, fmt)

        st.metric(label, display_value)

repetition_df = pd.DataFrame(
    {
        "Generation / Runtime Property": [
            "Mean generated tokens",
            "Generation throughput (samples/sec)",
            "Generation time (minutes)",
            "BERTScore computation time (seconds)",
        ],
        "Result": [
            f"{MODULE2_GENERATION_QUALITY['Mean generated tokens']:.6f}",
            f"{MODULE2_GENERATION_QUALITY['Generation throughput (samples/sec)']:.6f}",
            f"{MODULE2_GENERATION_QUALITY['Generation time (minutes)']:.6f}",
            f"{MODULE2_GENERATION_QUALITY['BERTScore time (seconds)']:.6f}",
        ],
    }
)

display_table(
    repetition_df,
    width="stretch",
    hide_index=True,
)


# ================================================================
# MODULE 3 — RETRIEVAL INDEX EVALUATION
# ================================================================

st.markdown(
    '<div class="section-header">7. Module 3 — Similar Image Retrieval</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="note">'
    "This section reports the completed index coverage and input data checks. "
    "These are index-readiness measures, not retrieval-accuracy scores. "
    "A relevance-labeled query set is needed to report Recall@K or mean reciprocal rank."
    "</div>",
    unsafe_allow_html=True,
)

if MODULE3_INDEX_EVALUATION is None:
    st.info(
        "No readable Module 3 index was found. Build it with "
        "`python -m src.module3.build_index` to show its coverage values here."
    )
else:
    module3 = MODULE3_INDEX_EVALUATION
    module3_cols = st.columns(4)
    module3_metrics = [
        ("Index Coverage", f"{module3['coverage']:.2%}"),
        ("Indexed Images", f"{module3['indexed']:,}"),
        ("Missing / Invalid Images", f"{module3['missing_images'] + module3['invalid_images']:,}"),
        ("Skipped ID / Text Rows", f"{module3['invalid_text_or_id']:,}"),
    ]
    for column, (label, value) in zip(module3_cols, module3_metrics):
        with column:
            st.metric(label, value)

    st.progress(
        module3["coverage"],
        text=f"{module3['indexed']:,} of {module3['total']:,} dataset rows indexed",
    )

    module3_data_quality = pd.DataFrame(
        [
            ["Dataset rows checked", f"{module3['total']:,}"],
            ["Images added to the index", f"{module3['indexed']:,}"],
            ["Index coverage", f"{module3['coverage']:.2%}"],
            ["Image files missing", f"{module3['missing_images']:,}"],
            ["Image files invalid", f"{module3['invalid_images']:,}"],
            ["Rows skipped for invalid/missing image_id or original text", f"{module3['invalid_text_or_id']:,}"],
            ["Saved index size", f"{module3['size_mb']:.2f} MiB"],
        ],
        columns=["Index / Data Check", "Result"],
    )
    display_table(
        module3_data_quality,
        width="stretch",
        hide_index=True,
    )

    module3_configuration = pd.DataFrame(
        [
            ["Visual encoder", module3["model"]],
            ["Execution device", "CPU"],
            ["Ranking method", "Cosine similarity over normalized image embeddings"],
            ["Maximum displayed matches", "5"],
            ["Report text source", "Original CSV text column (text_augment is not used)"],
        ],
        columns=["Retrieval Setting", "Value"],
    )
    display_table(
        module3_configuration,
        width="stretch",
        hide_index=True,
    )


# ================================================================
# EVALUATION SUMMARY
# ================================================================

st.markdown(
    '<div class="section-header">8. Evaluation Summary</div>',
    unsafe_allow_html=True,
)

summary_df = pd.DataFrame(
    [
        [
            "Module 1.1 — View Classification",
            "AP / Lateral classification",
            "Accuracy, Precision, Recall, F1",
            "0.9993 / 0.9993 / 0.9993 / 0.9993",
        ],
        [
            "Module 1 — Disease Prediction",
            "14-disease prediction",
            "AUROC, AP, F1, Accuracy, Precision, Recall, Specificity",
            "0.78919 / 0.75152 / 0.75001 / 0.84418 / "
            "0.70117 / 0.82407 / 0.48273",
        ],
        [
            "Module 2 — Report Generation",
            "Clinical report generation",
            "BLEU, ROUGE, BERTScore, TF-IDF cosine, repetition",
            "BLEU-1 0.244710 / BLEU-4 0.058915 / "
            "ROUGE-1 F1 0.338539 / ROUGE-L F1 0.229866 / "
            "BERTScore F1 0.831180 / Cosine mean 0.095690",
        ],
        [
            "Module 3 — Similar Image Retrieval",
            "Image index coverage and data validation",
            "Indexed coverage, missing/invalid images, skipped text/ID rows",
            (
                f"Coverage {MODULE3_INDEX_EVALUATION['coverage']:.2%} / "
                f"Indexed {MODULE3_INDEX_EVALUATION['indexed']:,} of "
                f"{MODULE3_INDEX_EVALUATION['total']:,}"
                if MODULE3_INDEX_EVALUATION
                else "Index not available"
            ),
        ],
    ],
    columns=[
        "Component",
        "Task",
        "Evaluation Metrics",
        "Selected Reported Results",
    ],
)

display_table(
    summary_df,
    width="stretch",
    hide_index=True,
)


# ================================================================
# FOOTER
# ================================================================

st.markdown(
    '<div class="footer">'
    "CXR Clinical AI · Verified Model Results and Saved Index Audit · Research Prototype"
    "</div>",
    unsafe_allow_html=True,
)
