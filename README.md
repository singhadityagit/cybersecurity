# 🛡️ ML Network Intrusion Detector

A beginner-friendly cybersecurity project that uses **Python, Streamlit, Pandas, and Scikit-learn** to explore labeled network-traffic datasets and train a machine-learning model to classify traffic records.

> **Note:** This is an educational prototype. It analyzes uploaded CSV data; it does not monitor live network traffic, block connections, or replace a production intrusion detection system.

## Features

- **Dashboard:** View dataset status, record count, and the latest model test accuracy.
- **Dataset Explorer:** Preview data, inspect column types, count missing values, and visualize selected columns.
- **Model Training:** Train a Random Forest classifier using a selected target-label column.
- **Model Evaluation:** View test accuracy, a confusion matrix, classification metrics, and feature importance.
- **Traffic Classification:** Select a record and view the model's predicted class and probability estimates.
- **Security Guide:** Learn about common network attack categories and basic security practices.
- **CSV Export:** Download a dataset preview or test predictions.

## Technologies

- Python
- Streamlit
- Pandas
- Scikit-learn
- Matplotlib

## Project Structure

```text
ML-Network-Detector/
├── app.py
├── README.md
├── requirements.txt
└── ML_Network_Intrusion_Detector_Project_Report.pdf   # optional report
```

The PDF report is optional and can be stored in this folder for project submission.

## Requirements

- Python 3.9 or later recommended
- Visual Studio Code (VS Code) or another Python editor
- A modern web browser
- A labeled network-traffic dataset in CSV format

## Installation and Setup (Windows / VS Code)

### 1. Open the project folder

Open `ML-Network-Detector` in VS Code, then choose **Terminal → New Terminal**.

### 2. Create a virtual environment

```powershell
python -m venv venv
```

### 3. Activate the environment

```powershell
venv\Scripts\activate
```

If PowerShell blocks activation, you can use Command Prompt instead, or run the application with the Python environment's executable directly.

### 4. Install dependencies

If `requirements.txt` exists:

```powershell
pip install -r requirements.txt
```

Otherwise, install the packages directly:

```powershell
python -m pip install streamlit pandas scikit-learn matplotlib
```

### 5. Run the application

Make sure the terminal is in the same folder as `app.py`, then run:

```powershell
streamlit run app.py
```

Open the local URL printed in the terminal (usually `http://localhost:8501`) in your browser.

## Prepare Your Dataset

1. Obtain a network-traffic dataset that you are permitted to use.
2. Save it as a CSV file.
3. Make sure it contains input feature columns and a target-label column.
4. Start the app and upload the CSV using the sidebar.
5. Open **Dataset Explorer** to review the columns and data quality.
6. Open **Train Model**, choose the correct target-label column, select a test percentage, and train the model.
7. Review accuracy, the confusion matrix, classification report, and feature importance.
8. Open **Detect Traffic** to classify a record.

Target labels vary by dataset. They may include values such as `Normal`, `Benign`, or attack names. Always verify what the labels mean before interpreting predictions.

## `requirements.txt`

You can create a `requirements.txt` file with the following contents:

```text
streamlit
pandas
scikit-learn
matplotlib
```

Then install dependencies with:

```powershell
pip install -r requirements.txt
```

## Evaluation Metrics

- **Accuracy:** Fraction of test records classified correctly.
- **Precision:** How often predictions for a class are correct.
- **Recall:** How many actual records of a class are identified.
- **F1-score:** A combined measure of precision and recall.
- **Confusion matrix:** Summarizes correct and incorrect predictions by class.
- **Feature importance:** Shows features the fitted Random Forest used more heavily in its decisions.

Do not rely on accuracy alone, especially when attack records are much less common than normal records.

## Limitations

- The app processes uploaded CSV files only; it does not capture live packets.
- Predictions depend on dataset quality, labels, selected features, and preprocessing.
- Dataset-specific formats may need additional cleaning.
- Probability estimates are not guarantees that traffic is safe or malicious.
- This prototype does not automatically block traffic or respond to incidents.

## Future Improvements

- Add dataset-specific preprocessing and validation.
- Use a preprocessing/model pipeline to reduce data leakage.
- Add cross-validation, precision-recall curves, and more evaluation charts.
- Save and reload trained models.
- Add live traffic capture only in an authorized, controlled lab.
- Add alerting and logging for a more advanced prototype.

## Responsible Use

Use only datasets and network information you are authorized to access. Treat model results as indicators for further investigation, not as definitive security findings.

## Author

- **Name:** [Your Name]
- **Roll Number:** [Your Roll Number]
- **Department / College:** [Your Department and College]
- **Project Guide:** [Guide Name]
- **Academic Year:** [Academic Year]

---

**Project:** ML Network Intrusion Detector  
**Purpose:** Academic learning and demonstration of machine-learning-based network traffic classification.
