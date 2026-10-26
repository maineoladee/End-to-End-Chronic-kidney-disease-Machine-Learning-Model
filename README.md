# Chronic Kidney Disease Prediction — Machine Learning & Flask

An end-to-end **machine learning demonstration** for predicting chronic kidney disease (CKD) from structured patient measurements. The repository brings together a Jupyter notebook for analysis and modelling, saved preprocessing/model artifacts, and a Flask interface for entering features and obtaining a prediction.

> **Portfolio project — not a medical device.** Predictions are for technical demonstration and learning, not for diagnosis, screening, or clinical decision-making. The model has not been established here as clinically validated.

## Project at a glance

| Component | Implementation |
| --- | --- |
| Task | Binary classification: CKD vs. not CKD |
| Language | Python |
| Modelling workflow | [`CKD.ipynb`](CKD.ipynb) |
| Inference application | [`app.py`](app.py) using Flask |
| Preprocessing artifact | [`scaling.pkl`](scaling.pkl) |
| Saved classifier | [`ckd_RFF.pkl`](ckd_RFF.pkl) |
| Interface | HTML templates in [`templates/`](templates/) and styling in [`static/css/`](static/css/) |
| Dependencies | [`requirements.txt`](requirements.txt) |

## What the application does

1. Displays a browser-based form for 24 patient measurements and clinical indicators.
2. Converts the submitted values into the numeric format expected by the application.
3. Loads the saved scaler and classifier and applies the scaler before inference.
4. Predicts one of two labels: **Chronic Kidney Disease** or **Not Chronic Kidney Disease**.
5. Displays the predicted class on the webpage.

The input fields cover age, blood pressure, urine findings, blood chemistry, blood counts, and medical-history indicators, including diabetes, hypertension and anaemia. The model expects the **same feature order and categorical coding used during training**. Consult [`app.py`](app.py) and the notebook before changing any inputs.

## Repository layout

```text
.
├── CKD.ipynb             # Exploratory analysis and model-development notebook
├── app.py                # Flask application and /predict endpoint
├── ckd_RFF.pkl           # Serialized trained model
├── scaling.pkl           # Serialized scaler
├── requirements.txt      # Pinned Python packages
├── Procfile              # Process configuration for web serving
├── templates/            # HTML views
├── static/css/           # Stylesheets
└── SPSS Output.spv       # Saved statistical output
```

## Run locally

Clone the repository:

```bash
git clone https://github.com/maineoladee/End-to-End-Chronic-kidney-disease-Machine-Learning-Model.git
cd End-to-End-Chronic-kidney-disease-Machine-Learning-Model
```

Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate
```

Install the supplied dependencies and start the Flask application:

```bash
python -m pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000/** in your browser and complete the form.

**Compatibility note:** The repository pins an older Python package environment, including Flask 2.1.2 and scikit-learn 1.1.1. A compatible Python interpreter and package versions may be necessary; the application has not been verified against current Python releases. The `.pkl` files should only be loaded from trusted sources, and upgrades to scikit-learn may require retraining or re-exporting the artifacts.

## Modelling and evaluation

[`CKD.ipynb`](CKD.ipynb) contains the project notebook. The Flask application loads the serialized classifier and scaler for inference, allowing the training workflow and browser-based prediction interface to be examined separately.

This README intentionally does **not** report an accuracy, F1-score, ROC-AUC, model ranking or external-validation result without a verified, reproducible source. Review the notebook to examine its analysis, modelling choices and recorded outputs. For a complete reproducibility assessment, confirm the dataset source, preprocessing decisions, train/test separation and version compatibility.

## Limitations and next steps

- **Clinical validity:** A working classifier is not evidence of prospective clinical safety or generalisability.
- **Input validation:** Add stricter checks on ranges, missing values and categorical encodings before wider use.
- **Reproducibility:** Document the dataset source and licensing, provide a clean training script and verify evaluation results on held-out data.
- **Software quality:** Add unit/integration tests, dependency updates, structured error handling and safer deployment configuration. The current `app.py` runs Flask with `debug=True` when started directly; that setting is for local development only.
- **Model monitoring:** A production implementation would need monitoring, security, access controls and formal validation.

## Skills demonstrated

**Python · scikit-learn · Flask · structured data preprocessing · serialized model inference · HTML/CSS · applied machine learning workflows**

## Author

[Oladayo Owoeye](https://github.com/maineoladee)
