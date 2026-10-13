"""Streamlit research demonstration: streamlit run app.py"""
from __future__ import annotations

import io
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

from mri_utils import CLASS_NAMES, load_image_array, read_json

PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACTS = Path(os.environ.get("MRI_ARTIFACTS_DIR", str(PROJECT_DIR / "artifacts"))).resolve()
MODEL_PATH = ARTIFACTS / "deployment" / "model.keras"
METADATA_PATH = ARTIFACTS / "deployment" / "metadata.json"

st.set_page_config(page_title="Brain MRI Classifier", page_icon="🧠", layout="wide")
st.title("Brain MRI Image Classifier")
st.caption("Educational research prototype • Four-class MRI image classification")
st.warning("This model is not clinically validated. Its predictions must not be used for diagnosis or treatment.")


@st.cache_resource
def load_assets(model_path, metadata_path, model_mtime, metadata_mtime):
    metadata = read_json(metadata_path)
    model = tf.keras.models.load_model(model_path, compile=False)
    if metadata.get("class_names") != CLASS_NAMES or model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError("Model class mapping does not match the four-class project.")
    size = int(metadata["image_size"])
    if tuple(model.input_shape[1:]) != (size, size, 3):
        raise ValueError("Model input size does not match deployment metadata.")
    return model, metadata


if not MODEL_PATH.is_file() or not METADATA_PATH.is_file():
    st.info("Train the models first, then place the generated artifacts folder beside app.py.")
    st.code("python brain_tumor_project.py --data dataset --output artifacts")
    st.stop()

try:
    model, metadata = load_assets(str(MODEL_PATH), str(METADATA_PATH),
                                 MODEL_PATH.stat().st_mtime_ns, METADATA_PATH.stat().st_mtime_ns)
except (OSError, ValueError, KeyError) as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

st.sidebar.header("Selected model")
st.sidebar.write(metadata["model_name"].replace("_", " ").title())
st.sidebar.caption("Selected using validation macro F1, before test evaluation.")
st.sidebar.write(f"Input size: {metadata['image_size']} × {metadata['image_size']}")
st.sidebar.caption("Only JPG or PNG brain MRI images are supported. DICOM is not supported.")

prediction_tab, results_tab = st.tabs(["Classify an MRI image", "Evaluation results"])
with prediction_tab:
    uploaded = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png"])
    st.caption("The classifier cannot verify that an upload is an MRI or reject all unfamiliar images.")
    if uploaded is not None:
        if uploaded.size > 20 * 1024 * 1024:
            st.error("Please upload an image smaller than 20 MB.")
        else:
            try:
                raw = uploaded.getvalue()
                with Image.open(io.BytesIO(raw)) as opened:
                    display_image = ImageOps.exif_transpose(opened).convert("RGB").copy()
                image = load_image_array(io.BytesIO(raw), int(metadata["image_size"]))
                left, right = st.columns([1, 1])
                with left:
                    st.image(display_image, caption="Uploaded MRI image", width="stretch")
                with right:
                    if st.button("Classify image", type="primary"):
                        with st.spinner("Running the model..."):
                            scores = np.asarray(model(image[None], training=False))[0]
                        index = int(scores.argmax())
                        predicted = metadata["class_names"][index].replace("_", " ").title()
                        st.subheader(f"Predicted image class: {predicted}")
                        st.metric("Highest model score", f"{scores[index]:.1%}")
                        values = pd.DataFrame({"Class": [c.replace("_", " ").title() for c in CLASS_NAMES],
                                               "Model score": scores.astype(float)})
                        st.bar_chart(values.set_index("Class"))
                        st.dataframe(values.assign(**{"Model score": values["Model score"].map(lambda x:f"{x:.2%}")}),
                                     hide_index=True, width="stretch")
                        st.caption("Scores are softmax outputs and are not calibrated clinical probabilities. "
                                   "A No Tumor prediction does not establish that a person is tumor-free.")
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                st.error(f"The image could not be processed: {exc}")

with results_tab:
    comparison = ARTIFACTS / "results" / "model_comparison.csv"
    if comparison.is_file():
        st.dataframe(pd.read_csv(comparison), hide_index=True, width="stretch")
        for filename, caption in [
            ("model_comparison.png", "Model comparison"),
            (f"{metadata['model_name']}_confusion_matrix.png", "Selected model test confusion matrix"),
            (f"{metadata['model_name']}_training_curves.png", "Selected model training history"),
        ]:
            path = ARTIFACTS / "results" / filename
            if path.is_file():
                st.image(str(path), caption=caption)
    else:
        st.info("Evaluation files were not included in this deployment.")
