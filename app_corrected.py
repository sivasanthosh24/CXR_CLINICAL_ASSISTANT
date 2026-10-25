from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent

VIEW_MODEL_DIR = PROJECT_ROOT / "models" / "module1_view"

RUNTIME_DIR = PROJECT_ROOT / ".runtime"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "app"

RUNTIME_DIR.mkdir(

    parents=True,

    exist_ok=True,

)

OUTPUT_DIR.mkdir(

    parents=True,

    exist_ok=True,

)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(0, str(PROJECT_ROOT))

from src.chatbot.controller import ChatbotController

st.set_page_config(

    page_title="CXR Clinical AI",

    page_icon="🩻",

    layout="wide",

    initial_sidebar_state="expanded",

)

try:

    from src.view_classifier.predictor import ViewClassifier

except Exception as import_error:

    st.error("Unable to import src.view_classifier.predictor.")

    st.exception(import_error)

    st.stop()


try:

    from src.module1.pipeline import run_module1_pipeline

except Exception as import_error:

    st.error("Unable to import src.module1.pipeline.")

    st.exception(import_error)

    st.stop()

if "chat_history" not in st.session_state:

    st.session_state.chat_history = [

        {

            "role": "assistant",

            "content": (

                "Welcome to CXR Clinical AI. Upload a chest X-ray "

                "and use the analysis controls in the sidebar."

            ),

        }

    ]

if "chatbot_history" not in st.session_state:

    st.session_state.chatbot_history = []

if "upload_widget_key" not in st.session_state:

    st.session_state.upload_widget_key = 0

if "uploaded_image" not in st.session_state:

    st.session_state.uploaded_image = None

if "temporary_image_path" not in st.session_state:

    st.session_state.temporary_image_path = None

if "uploaded_file_name" not in st.session_state:

    st.session_state.uploaded_file_name = None

if "view_result" not in st.session_state:

    st.session_state.view_result = None

if "module1_result" not in st.session_state:

    st.session_state.module1_result = None

if "module2_result" not in st.session_state:

    st.session_state.module2_result = None

if "view_inference_time" not in st.session_state:

    st.session_state.view_inference_time = None

if "module1_inference_time" not in st.session_state:

    st.session_state.module1_inference_time = None

if "caption_inference_time" not in st.session_state:

    st.session_state.caption_inference_time = None

if "show_gradcam" not in st.session_state:

    st.session_state.show_gradcam = False

if "show_bounding_box" not in st.session_state:

    st.session_state.show_bounding_box = False

def get_uploaded_image_path() -> Optional[Path]:

    image_path = st.session_state.get(

        "temporary_image_path"

    )

    if image_path is None:

        return None

    image_path = Path(image_path)

    if not image_path.is_file():

        return None

    return image_path

def append_chat_message(

    role: str,

    content: str,

) -> None:

    st.session_state.chat_history.append(

        {

            "role": role,

            "content": content,

        }

    )


def extract_module1_sections(

    module1_result: Dict[str, Any],

) -> Dict[str, Any]:

    prediction_result = module1_result.get(

        "prediction",

        {},

    )

    gradcam_result = module1_result.get(

        "gradcam",

        {},

    )

    localization_result = module1_result.get(

        "weak_localization",

        {},

    )

    positive_findings = prediction_result.get(

        "positive_findings",

        [],

    )

    ranked_findings = prediction_result.get(

        "ranked_findings",

        [],

    )

    selected_gradcam_disease = module1_result.get(

        "selected_gradcam_disease",

        gradcam_result.get(

            "explained_class_name",

            "Unknown",

        ),

    )

    return {

        "prediction": prediction_result,

        "gradcam": gradcam_result,

        "localization": localization_result,

        "positive_findings": positive_findings,

        "ranked_findings": ranked_findings,

        "selected_gradcam_disease": selected_gradcam_disease,

    }

def run_module2_caption_generation(

    image_path: Path,

    module1_result: Optional[Dict[str, Any]] = None,

) -> Dict[str, Any]:

    module2_pipeline = load_module2_pipeline()

    report = module2_pipeline.generate_report(

        image_path

    )

    model_info = module2_pipeline.get_model_info()

    return {

        "report": report,

        "narrative": report,

        "device": model_info.get(

            "device",

            str(module2_pipeline.device),

        ),

        "module": "Module 2 V2",

        "checkpoint": str(

            model_info.get(

                "checkpoint",

                "",

            )

        ),

    }

@st.cache_resource

def load_module2_pipeline():

    from src.module2.pipeline import load_pipeline

    return load_pipeline()

@st.cache_resource

def load_chatbot_controller() -> ChatbotController:

    return ChatbotController()

@st.cache_resource

def load_view_classifier() -> ViewClassifier:

    return ViewClassifier(

        model_directory=VIEW_MODEL_DIR,

    )

try:

    view_classifier = load_view_classifier()

except Exception as model_error:

    st.error(

        "Unable to load the trained AP/Lateral "

        "view classifier."

    )

    st.exception(model_error)
    st.stop()

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

    [data-testid="stSidebar"] {

        background: #111827;

    }

    [data-testid="stSidebar"] * {

        color: #e5e7eb;

    }

    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        min-height: 42px;
        border-radius: 9px;
        border: 1px solid #334155;
        background: #1f2937;
        color: #f8fafc;
        font-weight: 600;
        text-align: left;
    }
    [data-testid="stSidebar"] .stButton > button:hover {

        background: #273449;

        border-color: #64748b;
    }
    [data-testid="stSidebar"]

    .stButton > button[kind="primary"] {

        background: #2563eb;

        border-color: #3b82f6;

    }

    [data-testid="stSidebar"] .stFileUploader {

        background: #182235;

        border: 1px dashed #475569;

        border-radius: 10px;

        padding: 6px;

    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background: #2563eb !important;
    border: 1px solid #2563eb !important;
    color: #ffffff !important;
    border-radius: 8px;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {
    background: #1d4ed8 !important;
    border-color: #1d4ed8 !important;
    color: #ffffff !important;
}

    .app-title {

        font-size: 30px;

        font-weight: 750;

        color: #0f172a;

        margin-bottom: 2px;

    }

    .app-subtitle {

        color: #64748b;

        font-size: 14px;

        margin-bottom: 22px;

    }
    .chat-header {

        background: #0f172a;

        color: white;

        border-radius: 12px;

        padding: 15px 18px;

        margin-bottom: 12px;

    }

    .chat-header-title {

        font-size: 18px;

        font-weight: 700;

    }
    .chat-header-subtitle {
        color: #cbd5e1;
        font-size: 12px;
        margin-top: 3px;
    }
    .empty-chat {
        text-align: center;
        padding: 34px 20px;
        color: #64748b;
        background: #f8fafc;
        border: 1px dashed #cbd5e1;
        border-radius: 12px;
        margin-bottom: 14px;
    }
    .source-badge {
        display: inline-block;
        padding: 3px 8px;
        margin-top: 7px;
        margin-right: 5px;
        border-radius: 999px;
        background: #eff6ff;
        color: #1d4ed8;
        font-size: 11px;
    }
    </style>

    """,

    unsafe_allow_html=True,

)
# ================================================================

# SIDEBAR

# ================================================================

with st.sidebar:

    st.markdown(
        "### 🩻 CXR Clinical AI"
    )
    st.caption(
        "Chest X-ray analysis workspace"
    )
    st.markdown(
        "**INPUT IMAGE**"
    )
    uploaded_file = st.file_uploader(
        "Upload chest X-ray",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
        key=(
            f"xray_uploader_"
            f"{st.session_state.upload_widget_key}"
        ),
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        try:
            new_file = (
                st.session_state.get(
                    "uploaded_file_name"
                )
                != uploaded_file.name
            )
            uploaded_image = (
                Image.open(
                    uploaded_file
                ).convert("RGB")
            )
            if new_file:
                st.session_state.view_result = None
                st.session_state.module1_result = None
                st.session_state.module2_result = None
                st.session_state.view_inference_time = None
                st.session_state.module1_inference_time = None
                st.session_state.caption_inference_time = None
                st.session_state.show_gradcam = False
                st.session_state.show_bounding_box = False
                st.session_state.chatbot_history = []
            st.session_state.uploaded_image = (
                uploaded_image

            )
            st.session_state.uploaded_file_name = (

                uploaded_file.name

            )
            temporary_image_path = (

                RUNTIME_DIR / "uploaded_xray.jpg"

            )
            uploaded_image.save(

                temporary_image_path,

                format="JPEG",

            )
            st.session_state.temporary_image_path = (

                temporary_image_path

            )
        except Exception as image_error:
            st.error(

                "The uploaded image could not be read."

            )

            st.exception(image_error)

    if st.button(

        "Clear",

        use_container_width=True,

    ):

        st.session_state.uploaded_image = None
        st.session_state.uploaded_file_name = None
        st.session_state.temporary_image_path = None
        st.session_state.upload_widget_key += 1
        st.session_state.view_result = None
        st.session_state.module1_result = None
        st.session_state.module2_result = None
        st.session_state.view_inference_time = None
        st.session_state.module1_inference_time = None
        st.session_state.caption_inference_time = None
        st.session_state.show_gradcam = False
        st.session_state.show_bounding_box = False
        st.session_state.chat_history = []
        st.session_state.chatbot_history = []
        st.rerun()

    st.divider()

    st.markdown(

        "**ANALYSIS CONTROLS**"

    )

    image_path = get_uploaded_image_path()

    classify_view_button = st.button(

        "①  View Classification",

        use_container_width=True,

        type="primary",

        disabled=image_path is None,

    )

    disease_prediction_button = st.button(

        "②  Disease Prediction",

        use_container_width=True,

        type="primary",

        disabled=image_path is None,

    )

    gradcam_button = st.button(

        "③  Generate Grad-CAM",

        use_container_width=True,

        disabled=image_path is None,

    )

    bounding_box_button = st.button(

        "④  Show Bounding Box",

        use_container_width=True,

        disabled=image_path is None,

    )

    caption_button = st.button(

        "⑤  Generate Clinical Narrative",

        use_container_width=True,

        disabled=image_path is None,

    )
    st.divider()
    st.caption(
        f"Classifier device: "
        f"{view_classifier.device}"
    )
    st.caption(
        "Supported views: AP and Lateral"
    )
    st.caption(
        "PA classification: Not used"
    )
    st.caption(
        "Research prototype only."
    )

# ================================================================

# ANALYSIS HELPERS

# ================================================================

def execute_view_classification():
    current_path = (

        get_uploaded_image_path()

    )
    if current_path is None:

        return False

    if (

        st.session_state.get(

            "view_result"
        )
        is not None
    ):

        return True
    try:
        start = time.time()
        result = view_classifier.predict(
            current_path
        )
        st.session_state.view_result = result
        st.session_state.view_inference_time = (
            time.time() - start
        )
        return True
    except Exception as exc:
        st.error(
            "View classification failed."
        )
        st.exception(exc)
        return False
def execute_disease_prediction():
    current_path = (
        get_uploaded_image_path()
    )
    if current_path is None:
        return False
    if (
        st.session_state.get(
            "module1_result"
        )
        is not None
    ):
        return True
    if not execute_view_classification():
        return False
    predicted_view = (
        st.session_state.view_result.get(
            "predicted_view"
        )
    )
    if predicted_view not in {

        "AP",
        "Lateral",
    }:
        st.error(
            "Unsupported predicted view returned "
            "by the classifier."
        )
        return False
    try:
        start = time.time()
        try:
            result = run_module1_pipeline(
                image_source=current_path,
                view=predicted_view,
                output_dir=OUTPUT_DIR,
            )
        except TypeError:
            result = run_module1_pipeline(
                image_path=current_path,
                view=predicted_view,
                output_dir=OUTPUT_DIR,
            )
        st.session_state.module1_result = (
            result
        )
        st.session_state.module1_inference_time = (

            time.time() - start

        )
        return True
    except Exception as exc:
        st.error(
            "Disease prediction failed."
        )
        st.exception(exc)
        return False
def execute_module2():
    current_path = get_uploaded_image_path()
    if current_path is None:
        return False
    if st.session_state.get("module2_result") is not None:
        return True
    try:
        start = time.time()
        result = run_module2_caption_generation(current_path, None)
        if isinstance(result, dict):
            raw_narrative = (
                result.get("narrative")
                or result.get("report")
                or result.get("caption")
                or result.get("clinical_caption")
                or result.get("generated_text")
                or result.get("text")
            )
        else:
            raw_narrative = str(result)
        if raw_narrative:
            controller = load_chatbot_controller()
            refinement_result = controller.narrative_refiner.refine(
                str(raw_narrative)
            )
            result = dict(result) if isinstance(result, dict) else {"report": str(result)}
            result["raw_narrative"] = str(raw_narrative)
            result["narrative"] = (
                refinement_result.refined_narrative
                or str(raw_narrative)
            )
            result["refinement"] = {
                "success": refinement_result.success,
                "model": refinement_result.model,
                "error": refinement_result.error,
            }
        st.session_state.module2_result = result
        st.session_state.caption_inference_time = time.time() - start
        return True
    except Exception as exc:
        st.error("Module 2 V2 narrative generation failed.")
        st.exception(exc)
        return False


def prepare_chat_analysis(

    question: str,
):
    q = question.lower().strip()
    if any(
        x in q

        for x in [
            "grad-cam",
            "gradcam",
            "heatmap",
            "activation",
        ]

    ):

        if execute_disease_prediction():
            st.session_state.show_gradcam = True
    elif any(

        x in q

        for x in [
            "bounding box",
            "localization",
            "localized",
            "abnormal region",
        ]
    ):
        if execute_disease_prediction():
            st.session_state.show_bounding_box = True
    elif any(
        x in q
        for x in [
            "narrative",
            "caption",
            "report",
            "clinical narrative",
            "generate caption",
        ]
    ):
        execute_module2()
    elif any(
        x in q
        for x in [
            "disease",
            "probability",
            "probabilities",
            "abnormalit",
            "prediction",
        ]
    ):
        execute_disease_prediction()
    elif any(
        x in q
        for x in [
            "view",
            "ap",
            "lateral",
        ]
    ):
        execute_view_classification()
    elif any(
        x in q
        for x in [
            "summarize",
            "summary",
            "overall",
            "all results",
            "findings",
        ]
    ):
        execute_module2()

# ================================================================
# SIDEBAR BUTTON ACTIONS
# ================================================================

if classify_view_button:

    with st.spinner(

        "Identifying the X-ray view\..."

    ):

        execute_view_classification()
if disease_prediction_button:
    with st.spinner(
        "Running disease prediction..."
    ):
        execute_disease_prediction()

if gradcam_button:

    if execute_disease_prediction():

        st.session_state.show_gradcam = True
if bounding_box_button:
    if execute_disease_prediction():
        st.session_state.show_bounding_box = True
if caption_button:
    with st.spinner(
        "Generating the Module 2 V2 "
        "clinical narrative..."
    ):
        execute_module2()

# ================================================================
# MAIN HEADER
# ================================================================
st.title(
    "🩻 CXR Clinical AI"
)

st.caption(

    "Multimodal chest X-ray analysis · "

    "Module 1 + Module 2  · "

    "Local clinical assistant chatbot"
)

# ================================================================
# ANALYSIS OUTPUT
# ================================================================

st.subheader(

    "Analysis Output"

)

st.caption(
    "All available results for the current chest X-ray "
    "are shown here."
)

with st.container(

    border=True

):

    uploaded_image = (
        st.session_state.get(
            "uploaded_image"
        )
    )

    if uploaded_image is None:

        st.info(

            "Upload a chest X-ray from the sidebar to begin."

        )

    else:

        image_col, summary_col = st.columns(

            [1.05, 0.95],

            gap="large",

        )

        with image_col:

            st.markdown(

                "**Input X-ray**"

            )

            st.image(

                uploaded_image,

                width=440,

            )

            st.caption(
                st.session_state.get(
                    "uploaded_file_name",
                    "Uploaded image",

                )

            )

        with summary_col:

            st.markdown(
                "**Current analysis status**"
            )

            view_result = (
                st.session_state.get(
                    "view_result"
                )
            )
            module1_result = (
                st.session_state.get(
                    "module1_result"
                )
            )
            module2_result = (
                st.session_state.get(
                    "module2_result"
                )
            )

            status_rows = [
                (
                    "View Classification",
                    "Ready"
                    if view_result
                    else "Not run",
                ),
                (
                    "Disease Probabilities",
                    "Ready"
                    if module1_result
                    else "Not run",
                ),
                (
                    "Grad-CAM",
                    "Ready"
                    if (
                        st.session_state.get(
                            "show_gradcam"
                        )
                        and module1_result
                    )
                    else "Not run",
                ),
                (
                    "Weak Localization",
                    "Ready"
                    if (
                        st.session_state.get(
                            "show_bounding_box"
                        )
                        and module1_result
                    )
                    else "Not run",
                ),
                (
                    "Clinical Narrative",
                    "Ready"
                    if module2_result
                    else "Not run",
                ),
            ]
            for label, status in status_rows:
                st.write(
                    f"**{label}:** {status}"
                )
        view_result = (
            st.session_state.get(
                "view_result"
            )
        )
        if view_result:
            st.divider()
            st.markdown(
                "**View Classification**"
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(
                    "Predicted View",
                    str(
                        view_result.get(
                            "predicted_view",
                            "Unknown",
                        )
                    ),
                )
            with c2:
                st.metric(
                    "Confidence",
                    (
                        f"{float(view_result.get('confidence', 0.0)) * 100:.2f}%"
                    ),
                )
            with c3:
                value = (
                    st.session_state.get(
                        "view_inference_time"
                    )
                )
                st.metric(
                    "Inference",
                    (
                        f"{value * 1000:.0f} ms"
                        if value
                        else "—"
                    ),
                )
        module1_result = (
            st.session_state.get(
                "module1_result"
            )
        )
        module1_sections = (
            extract_module1_sections(
                module1_result
            )
            if module1_result
            else None
        )

        if module1_sections:
            ranked = (
                module1_sections.get(
                    "ranked_findings",
                    [],
                )
            )
            if ranked:
                st.divider()
                st.markdown(
                    "**Disease Probabilities**"
                )
                rows = []
                for rank, item in enumerate(
                    ranked,
                    1,
                ):
                    name = item.get(
                        "disease",
                        item.get(
                            "class_name",
                            "Unknown",
                        ),
                    )
                    try:
                        probability = float(
                            item.get(
                                "probability",
                                item.get(
                                    "score",
                                    0.0,
                                ),
                            )
                        )
                    except Exception:
                        probability = 0.0
                    rows.append(

                        {
                            "Rank": rank,
                            "Disease": name,
                            "Probability": (
                                f"{probability * 100:.2f}%"
                            ),
                        }
                    )

                st.dataframe(
                    rows,
                    use_container_width=True,
                    hide_index=True,
                )
        if (
            st.session_state.get(
                "show_gradcam"
            )
            and module1_sections
        ):
            st.divider()
            st.markdown(
                "**Grad-CAM Explanation**"
            )
            gradcam = (
                module1_sections.get(
                    "gradcam",
                    {},
                )
            )
            g1, g2, g3 = st.columns(3)
            with g1:
                st.metric(
                    "Explained Disease",
                    str(
                        gradcam.get(
                            "explained_class_name",
                            module1_sections.get(
                                "selected_gradcam_disease",
                                "Unknown",
                            ),
                        )
                    ),
                )
            with g2:
                st.metric(
                    "Predicted Class",
                    str(
                        gradcam.get(
                            "predicted_class_name",
                            "Unknown",
                        )
                    ),
                )
            with g3:
                st.metric(
                    "Target Layer",
                    str(
                        gradcam.get(
                            "target_layer",
                            "Unknown",
                        )
                    ),
                )
            paths = gradcam.get(
                "output_paths",
                {},
            )
            images = [
                (
                    "Combined",
                    paths.get("combined"),
                ),
                (
                    "Overlay",
                    paths.get("overlay"),
                ),
                (
                    "Heatmap",
                    paths.get("heatmap"),
                ),
            ]
            valid = [
                (
                    title,
                    Path(path),
                )
                for title, path in images
                if path
                and Path(path).is_file()
            ]
            if valid:
                cols = st.columns(
                    len(valid)
                )
                for col, (
                    title,
                    path,
                ) in zip(
                    cols,
                    valid,
                ):
                    with col:
                        st.caption(
                            title
                        )
                        st.image(
                            str(path),
                            use_container_width=True,
                        )
        if (
            st.session_state.get(
                "show_bounding_box"
            )
            and module1_sections
        ):
            st.divider()
            st.markdown(

                "**Weak Localization / Bounding Box**"
            )
            localization = (
                module1_sections.get(
                    "localization",
                    {},
                )
            )
            available = localization.get(
                "localization_available",
                False,
            )
            if available:
                b1, b2 = st.columns(2)
                with b1:
                    area = (
                        float(
                            localization.get(
                                "component_area_ratio",
                                0.0,
                            )
                        )
                        * 100
                    )
                    st.metric(
                        "Activated Area",
                        f"{area:.2f}%",
                    )
                with b2:
                    st.write(
                        "**Approximate Center**"
                    )
                    st.json(
                        localization.get(
                            "center"
                        )
                        or {}
                    )
                st.write(
                    "**Approximate Bounding Box**"
                )
                st.json(
                    localization.get(
                        "bounding_box"
                    )
                    or {}
                )
            else:
                st.warning(
                    "No sufficiently large activated region "
                    "was identified by the weak-localization "
                    "procedure."
                )
            viz = localization.get(
                "visualization_path"
            )
            if (
                viz
                and Path(viz).is_file()
            ):
                st.image(
                    str(viz),
                    width=500,
                )
        module2_result = (
            st.session_state.get(
                "module2_result"
            )
        )
        if module2_result:
            st.divider()
            st.markdown(
                "**Module 2 Clinical Narrative**"
            )
            if isinstance(
                module2_result,
                str,
            ):
                narrative = module2_result
            else:
                narrative = (
                    module2_result.get(
                        "narrative"
                    )
                    or module2_result.get(
                        "report"
                    )
                    or module2_result.get(
                        "caption"
                    )
                    or module2_result.get(
                        "clinical_caption"
                    )
                    or module2_result.get(
                        "generated_text"
                    )
                    or module2_result.get(
                        "text"
                    )
                )
            if narrative:
                st.write(
                    narrative
                )
            else:
                st.json(
                    module2_result
                )

# ================================================================
# CHATBOT EVIDENCE RENDERING
# ================================================================
def render_chat_evidence(
    category: str,
):
    module1_result = (
        st.session_state.get(
            "module1_result"
        )
    )
    module1_sections = (
        extract_module1_sections(
            module1_result
        )
        if module1_result
        else None
    )
    if (
        category == "disease_prediction"
        and module1_sections
    ):
        ranked = (
            module1_sections.get(
                "ranked_findings",
                [],
            )
        )
        if ranked:
            rows = []
            for rank, item in enumerate(
                ranked,
                1,
            ):
                name = item.get(
                    "disease",
                    item.get(
                        "class_name",
                        "Unknown",
                    ),
                )
                try:
                    probability = float(
                        item.get(
                            "probability",
                            item.get(
                                "score",
                                0.0,
                            ),
                        )
                    )
                except Exception:
                    probability = 0.0
                rows.append(
                    {
                        "Rank": rank,
                        "Disease": name,
                        "Probability": (
                            f"{probability * 100:.2f}%"
                        ),
                    }
                )
            st.dataframe(
                rows,
                use_container_width=True,
                hide_index=True,
            )
    elif (
        category == "gradcam"
        and module1_sections
    ):
        gradcam = (
            module1_sections.get(
                "gradcam",
                {},
            )
        )
        paths = gradcam.get(
            "output_paths",
            {},
        )
        valid = [
            (
                title,
                Path(path),
            )
            for title, path in [
                (
                    "Combined",
                    paths.get("combined"),
                ),
                (
                    "Overlay",
                    paths.get("overlay"),
                ),
                (
                    "Heatmap",
                    paths.get("heatmap"),
                ),
            ]
            if path
            and Path(path).is_file()
        ]
        if valid:
            cols = st.columns(
                len(valid)
            )
            for col, (
                title,
                path,
            ) in zip(
                cols,
                valid,
            ):
                with col:
                    st.caption(
                        title
                    )
                    st.image(
                        str(path),
                        use_container_width=True,
                    )
    elif (
        category == "localization"
        and module1_sections
    ):
        localization = (
            module1_sections.get(
                "localization",
                {},
            )
        )
        if localization.get(
            "localization_available",
            False,
        ):
            st.write(
                "**Approximate Bounding Box**"
            )
            st.json(
                localization.get(
                    "bounding_box"
                )
                or {}
            )
            viz = localization.get(
                "visualization_path"
            )
            if (
                viz
                and Path(viz).is_file()
            ):
                st.image(
                    str(viz),
                    width=500,
                )
        else:
            st.warning(
                "No sufficiently large activated "
                "region was identified by the "
                "weak-localization procedure."
            )
    elif (
        category == "view"
        and st.session_state.get(
            "view_result"
        )
    ):
        result = (
            st.session_state.view_result
        )
        c1, c2 = st.columns(2)
        with c1:
            st.metric(
                "Predicted View",
                str(
                    result.get(
                        "predicted_view",
                        "Unknown",
                    )
                ),
            )
        with c2:
            st.metric(
                "Confidence",
                (
                    f"{float(result.get('confidence', 0.0)) * 100:.2f}%"
                ),
            )
    elif (
        category == "narrative"
        and st.session_state.get(
            "module2_result"
        )
    ):
        result = (
            st.session_state.module2_result
        )
        if isinstance(
            result,
            str,
        ):
            narrative = result
        else:
            narrative = (
                result.get(
                    "report"
                )
                or result.get(
                    "narrative"
                )
                or result.get(
                    "caption"
                )
                or result.get(
                    "clinical_caption"
                )
                or result.get(
                    "generated_text"
                )
                or result.get(
                    "text"
                )
            )

        if narrative:

            st.info(
                narrative
            )
# ================================================================
# PRIMARY CHATBOT
# ================================================================
st.subheader(
    "CXR Clinical Assistant"
)
st.caption(
    "Ask a focused question. The assistant uses the available "
    "module outputs and can run the required analysis automatically."

)
with st.container(
    border=True
):
    st.markdown(
        "**Clinical Analysis Chat**"
    )
    st.caption(
        "Local CXR asistant "
    )
    if not st.session_state.chatbot_history:
        st.markdown(
            "**Start here**"
        )
        st.caption(
            "Try: What is the predicted view? · "
            "What are the disease probabilities? · "
            "Show the Grad-CAM result · "
            "Show the bounding box · "
            "Summarize the clinical narrative."
        )
    for message in (
        st.session_state.chatbot_history
    ):
        role = message.get(
            "role",
            "assistant",
        )
        with st.chat_message(
            role
        ):
            st.markdown(
                message.get(
                    "content",
                    "",
                )
            )
            if role == "assistant":
                sources = message.get(
                    "source_modules",
                    [],
                )
                if sources:
                    st.caption(
                        "Sources: "
                        + " · ".join(
                            sources
                        )
                    )
                render_chat_evidence(
                    message.get(
                        "category",
                        "general",
                    )
                )
    with st.form(
        "clinical_chat_form",
        clear_on_submit=True,
    ):
        chat_question = st.text_input(
            "Ask about the current CXR",
            placeholder=(
                "e.g. What are the disease probabilities?"
            ),
            label_visibility="collapsed",
        )
        send = st.form_submit_button(
            "Send",
            use_container_width=True,
        )
if (
    send
    and chat_question.strip()
):
    question = (
        chat_question.strip()
    )
    st.session_state.chatbot_history.append(
        {
            "role": "user",
            "content": question,
        }
    )
    with st.spinner(
        "Analyzing the request..."
    ):
        prepare_chat_analysis(
            question
        )
        controller = (
            load_chatbot_controller()
        )
        response = controller.ask(
            question=question,
            view_result=st.session_state.get(
                "view_result"
            ),
            module1_result=st.session_state.get(
                "module1_result"
            ),
            module2_result=st.session_state.get(
                "module2_result"
            ),
        )
    display_category = (
        response.metadata.get("evidence_intent")
        if isinstance(response.metadata, dict)
        else None
    ) or response.category
    st.session_state.chatbot_history.append(
        {
            "role": "assistant",
            "content": response.text,
            "source_modules": response.source_modules,
            "category": display_category,
        }
    )
    st.rerun()
st.markdown("---")
st.caption(
    "CXR Clinical AI · Module 1 + Module 2 V2 · Prototype"
)