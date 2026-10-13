# Brain Tumor MRI Image Classification

This project trains a custom CNN, MobileNetV2 and EfficientNetB0 on the assigned
four-class MRI dataset, compares them, and deploys the validation-selected model
in Streamlit. Labels are glioma, meningioma, no_tumor and pituitary.

**Package status:** source code and the Colab notebook are supplied. The real MRI
dataset and trained weights are not included in the source package. Run training
before submitting a completed project. The scripts compute metrics from your
run; no accuracy figures or trained-model claims are prefilled.

## Fastest route for today's submission

1. Open `Brain_Tumor_MRI_Colab.ipynb` in Google Colab using **File → Upload notebook**.
2. Select **Runtime → Change runtime type → T4 GPU** (or another available GPU).
3. Download the complete dataset folder from the assignment link as a ZIP:
   https://drive.google.com/drive/folders/1C9ww4JnZ2sh22I-hbt45OR16o4ljGxju?usp=sharing
4. Run the notebook cells in order. When asked for a file, upload the **dataset ZIP**.
   An optional Drive shortcut route is included for larger files.
5. The final cell downloads `Brain_Tumor_MRI_Trained_Submission.zip` containing
   code, three `.h5` trained models, the selected Streamlit model, metadata,
   computed metrics, figures and `REPORT.md`.
6. Download the executed notebook through **File → Download → Download .ipynb**
   and submit it alongside the trained ZIP if notebooks are requested.
7. Upload the source and generated results to your own public GitHub repository
   and include its link, as required by the assignment.

Training time depends on GPU availability and the selected epoch counts.
The default run uses 12 initial epochs and up to 3 fine-tuning epochs per
transfer model. Early stopping can finish sooner. It does not guarantee a score.
You can use 8 initial epochs and 2 fine-tuning epochs when time is limited;
the quality of that run must still be judged from its actual validation results.

## Files in this package

| File | Purpose |
|---|---|
| `brain_tumor_project.py` | Train, clean data, evaluate, compare and export all models |
| `mri_utils.py` | Shared image decoding, class mapping and safe ZIP extraction |
| `app.py` | Streamlit image upload, class scores and evaluation display |
| `make_submission.py` | Validate required outputs and build the trained submission ZIP |
| `Brain_Tumor_MRI_Colab.ipynb` | GPU notebook with embedded project scripts |
| `requirements.txt` | Local project dependencies |
| `requirements-colab.txt` | Training dependencies for Colab |
| `DATASET_INFO.json` | Verified source split/class counts before cleaning |
| `SUBMISSION_CHECKLIST.md` | Required deliverables and submission steps |

## Dataset layout

Extract the entire assigned folder, preserving these directories:

- `dataset/train/glioma`, `meningioma`, `no_tumor`, `pituitary`
- `dataset/valid/glioma`, `meningioma`, `no_tumor`, `pituitary`
- `dataset/test/glioma`, `meningioma`, `no_tumor`, `pituitary`

Each split has the four class folders. `validation` and `val` are accepted
alternatives to `valid`. Extra ZIP wrapper directories are detected automatically.
The supplied `_classes.csv` files are not required by the loader because the
class folder already supplies each label. DICOM files are not supported.

## Local training

Use Python 3.10–3.12 with a compatible TensorFlow installation. A Colab GPU is
recommended for completing all three models. CPU training on a laptop can take
substantially longer. From the extracted project directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python brain_tumor_project.py --data dataset --output artifacts
python make_submission.py --artifacts artifacts
streamlit run app.py
```

On Windows, activate the environment using `.venv\Scripts\activate`.

For a shorter real run:

```bash
python brain_tumor_project.py --data dataset --output artifacts --epochs 8 --fine-tune-epochs 2
```

If GPU memory is insufficient, add `--batch-size 16`. A new training run must use
a new or empty output directory. This protects against mixing old and new metrics.
The default 224-pixel images are appropriate for the pretrained backbones.
Internet access is needed for the first download of ImageNet weights.

## Data integrity and evaluation

The assignment's CSV inventories report 1,695 training, 502 validation and 246
test images. Their class counts are recorded in `DATASET_INFO.json`.
Source prefixes before `.rf.` overlap across the splits: 97 distinct prefixes
between training and validation, 57 between training and test, and 10 between
validation and test. These three overlap counts must not be added because a
family can appear in all three splits.

The pipeline groups exact decoded-pixel duplicates and Roboflow source-name
families. It retains one example per family, prioritizing test, then validation,
then training. Conflicting-label families and unreadable files are excluded and
logged. The retained-image manifest records the actual image dimensions and
labels. Cleaned counts differ from the original counts. Files in the source
dataset are never modified or deleted.

Only training images are augmented. Class weights are calculated from training
labels only. The selected model has the highest validation macro F1; validation
accuracy and fewer parameters break ties. Selection happens before test
predictions are computed. All three models are then evaluated on the held-out
test split for comparison. Test results must not be used for repeated tuning.

Normalization is part of the saved model, avoiding training/deployment mismatch:

| Model | Input to the saved model | Internal preprocessing |
|---|---|---|
| Custom CNN | RGB pixels 0–255 | Scale to 0–1 |
| MobileNetV2 | RGB pixels 0–255 | Scale to −1–1 |
| EfficientNetB0 | RGB pixels 0–255 | Built-in EfficientNet preprocessing |

Training and Streamlit both use EXIF-corrected RGB decoding and PIL bilinear
resizing. Transfer learning starts with ImageNet weights and a frozen backbone.
Optional fine-tuning unfreezes the last layers while keeping batch normalization
frozen and using a learning rate of 1e-5.

## Generated outputs

- `artifacts/models/`: `.h5` and native `.keras` exports for each trained model.
- `artifacts/deployment/`: selected `model.keras` and `metadata.json`.
- `artifacts/results/`: class distribution and sample images, training curves,
  history CSVs, per-model validation and test metrics, confusion matrices,
  test predictions, model comparison CSV/JSON/PNG, and `REPORT.md`.
- `artifacts/class_names.json`: fixed output-label order.
- `artifacts/COMPLETED.json`: completion status for the packaging check.

The trained ZIP includes the three required `.h5` exports plus the selected
native `.keras` deployment model. It omits redundant per-model `.keras` copies
and the MRI dataset to keep the upload smaller. TensorFlow considers HDF5 a
legacy format; the project includes it because the brief explicitly requests it.

## Run the application after Colab training

Extract the **trained** ZIP on your computer. Its project directory contains
`app.py` and `artifacts/` together. Install `requirements.txt` and run:

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.
The app displays a training instruction when trained weights are absent.
To use a different artifacts directory, set the `MRI_ARTIFACTS_DIR` environment
variable to its absolute path before starting Streamlit.

## GitHub and submission

Create a public GitHub repository using your own account. Upload these source
files and the notebook, plus useful results such as comparison figures and
metrics. Follow the dataset's licensing terms before redistributing any images.
The supplied `.gitignore` keeps model weights and dataset images out of an
ordinary Git commit. Upload the trained ZIP to the assignment portal; for model
files on GitHub use a release asset or Git LFS if repository limits require it.
No repository has been created or published by this package.

## Scope and limitations

This is an educational MRI image classifier. It has not been clinically
validated and cannot diagnose a patient, establish that a person is tumor-free,
or reliably reject non-MRI/out-of-distribution uploads. The displayed softmax
scores are not calibrated clinical probabilities. There are no supplied patient
IDs, so patient-level independence cannot be verified; the grouping method does
not detect every near duplicate. Performance on this curated dataset does not
establish performance at another hospital or with a different MRI protocol.

`--weights none` is reserved for software verification with synthetic data.
It does not implement pretrained transfer learning. `make_submission.py` refuses
to package such runs as final assignment results.

## Dataset attribution and references

The dataset README identifies **Labeled MRI Brain Tumor Dataset Version 1**,
Roboflow user Ali Rostami, and **CC BY 4.0**. Its labeling claims have not been
independently verified. Cleaning, preprocessing, and source grouping described
above are transformations performed by this project.

- Original dataset page: https://universe.roboflow.com/ali-rostami/labeled-mri-brain-tumor-dataset
- Dataset README: https://drive.google.com/file/d/1H8DhtZjx4bilk-6kebL6087Xu0TVxv33/view?usp=drivesdk
- Keras transfer-learning guide: https://keras.io/guides/transfer_learning/
- Keras MobileNet models: https://keras.io/api/applications/mobilenet/mobilenet_models/
- Keras EfficientNet models: https://keras.io/api/applications/efficientnet/efficientnet_models/
- Streamlit resource caching: https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource

## Software verification

The delivered code was checked with Python 3.12, TensorFlow CPU 2.19.1 and
Streamlit 1.50.0. A short synthetic-image run exercised all three models,
fine-tuning, evaluation, figures, native checkpoints, HDF5 exports and selected
deployment creation. Native/HDF5 predictions matched for each model. Additional
checks covered duplicate/source-family isolation, conflicting labels, ZIP path
traversal rejection, notebook syntax and embedded-source consistency, and app
startup both with and without generated model files. The packager rejected
random-weight software-test results.

These checks verify software behavior, not MRI classification performance.
No synthetic models, synthetic metrics or synthetic sample images are included
in this source package. Full MRI training, GPU execution and clinical validation
have not been performed here.
