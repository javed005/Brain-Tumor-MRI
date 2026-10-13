# Submission checklist

## Run today

1. Upload `Brain_Tumor_MRI_Colab.ipynb` to Colab and select a GPU runtime.
2. Run setup and upload the complete assigned dataset ZIP when asked.
3. Run the full three-model training cell with ImageNet weights.
4. Confirm that all three comparison rows and their computed test results appear.
5. Run the export cell and download `Brain_Tumor_MRI_Trained_Submission.zip`.
6. Download the executed `.ipynb` separately from Colab's File menu.
7. Publish your code to a public GitHub repository and copy its URL.
8. Test the Streamlit application locally after extracting the trained ZIP.

## Upload to the assignment portal

- Trained submission ZIP containing scripts, trained `.h5` models, selected
  deployment model, metadata, actual metrics/charts and generated `REPORT.md`.
- Executed notebook, if the portal requests a separate notebook file.
- Your public GitHub repository URL, as required by the project brief.
- Screenshots or a video only if your evaluator separately requires them.

The provided source ZIP alone does not contain trained models or measured MRI
performance. Do not submit random-weight software-test outputs as final results.
There is no supplied test score until the real-data notebook has finished.

## Check the trained archive

- `artifacts/models/custom_cnn.h5`
- `artifacts/models/mobilenet_v2.h5`
- `artifacts/models/efficientnet_b0.h5`
- `artifacts/deployment/model.keras`
- `artifacts/deployment/metadata.json`
- `artifacts/results/model_comparison.csv`
- `artifacts/results/REPORT.md`
- Per-model test metrics, confusion matrices and training curves.
- `artifacts/COMPLETED.json`

Keep the test set for final evaluation. The app and report must identify this as
an educational prototype without clinical validation.
