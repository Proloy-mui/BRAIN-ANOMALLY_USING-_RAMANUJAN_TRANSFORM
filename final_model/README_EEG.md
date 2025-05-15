
# 🧠 EEG Signal Classification using DFT, DWT, and Ramanujan Transform

## 📘 Overview

This project focuses on classifying EEG (Electroencephalogram) signals by extracting signal features using Discrete Fourier Transform (DFT), Discrete Wavelet Transform (DWT), and a placeholder for the Ramanujan Transform. The extracted features are fed into ensemble machine learning models to detect patterns and classify neurological states or events.

---

## 🧰 Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - `numpy`, `pandas` – data manipulation
  - `scikit-learn` – preprocessing, modeling, evaluation
  - `xgboost` – gradient boosting machine learning model
  - `pywavelets` – wavelet transformation
  - `scipy` – DFT via FFT
  - `matplotlib` – visualizations
  - `imbalanced-learn` – handling class imbalance (optional)

---

## 📂 Dataset

- **File**: `eeg_dataset.csv`
- **Structure**: Each row represents an EEG signal. All columns except the last are signal values. The last column is the target label for classification.

---

## 📈 Feature Extraction

For each EEG signal:

1. **DFT (Discrete Fourier Transform)**:
   - Captures frequency domain features using FFT.
   - Only the first half of coefficients (positive frequencies) are retained.

2. **DWT (Discrete Wavelet Transform)**:
   - Performed using the Haar wavelet up to level 3 decomposition.
   - Wavelet coefficients are concatenated to represent the signal.

3. **Ramanujan Transform (Placeholder)**:
   - A simple approximation using `log1p(abs(signal))` to simulate periodic pattern detection.
   - Actual mathematical transform is not implemented.

4. **Concatenation**:
   - All three transformations are combined into one comprehensive feature vector per EEG signal.

---

## 🤖 Machine Learning Pipeline

1. **Data Splitting**:
   - Dataset is split into training and test sets.

2. **Scaling**:
   - Features are standardized using `StandardScaler`.

3. **Models Used**:
   - `RandomForestClassifier`
   - `XGBClassifier`
   - Both are combined using a `VotingClassifier` for ensemble learning.

4. **Evaluation**:
   - Accuracy score
   - Confusion matrix
   - Classification report (precision, recall, F1-score)

---

## 🧪 Testing

After training, the model is tested on unseen EEG signals using the test split. Evaluation metrics include:

- **Accuracy**: Overall correct classification rate.
- **Confusion Matrix**: Counts of true positives, false positives, etc.
- **Classification Report**: Provides precision, recall, F1-score for each class.

```python
# Example from the notebook
y_pred = voting_clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
```

---

## 📊 Visualization

The notebook includes plots to visualize the following transforms on a sample EEG signal:
- **DFT Spectrum** (Frequency vs Magnitude)
- **DWT Coefficients** (Wavelet scale patterns)
- **Ramanujan Transform Output** (Log-transformed signal magnitude)

---

## 🛠 Installation & Running

1. **Install Dependencies**:
   ```bash
   pip install numpy pandas scikit-learn xgboost imbalanced-learn pywavelets
   ```

2. **Run Notebook**:
   - Open `EEGfinal.ipynb` in Jupyter Notebook or JupyterLab.
   - Ensure `eeg_dataset.csv` is correctly placed and the path is set properly in the code.
   - Execute all cells in sequence.

---

## 🚀 Future Improvements

- Implement the actual **Ramanujan Periodicity Transform** for more accurate time-series representation.
- Add **cross-validation** to reduce evaluation bias.
- Integrate **deep learning models** (CNN, LSTM) for complex feature detection.
- Perform **dimensionality reduction** (e.g., PCA) to optimize training time.
- Handle **imbalanced datasets** using SMOTE or similar techniques.

---

## 🧠 Authors

- @Proloy_mui

---

## 📌 License

This project is open for educational and research use. Please cite the source if used in publications.
