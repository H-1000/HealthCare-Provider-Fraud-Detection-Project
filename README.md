# Healthcare Provider Fraud Detection

A machine learning project to detect fraudulent healthcare providers using Medicare claims data. This project implements multiple classification algorithms with SMOTE oversampling to handle class imbalance and provides comprehensive evaluation metrics.

## 📊 Project Overview

Healthcare fraud is a significant problem costing billions of dollars annually. This project analyzes Medicare claims data to identify potentially fraudulent healthcare providers based on their billing patterns, claim characteristics, and patient demographics.

## Team Members:

Omar Bassem Talaat Mustafa
Abdelhamid Taher Elnaggar
Adham Bahaa Barakat
Sherif Ahmed Halawa


### Key Features
- **Multi-source data integration**: Combines beneficiary, inpatient, and outpatient data
- **Feature engineering**: Aggregates claim-level data to provider-level features
- **Class imbalance handling**: Uses SMOTE (Synthetic Minority Over-sampling Technique)
- **Multiple model comparison**: Evaluates Logistic Regression, Random Forest, and Gradient Boosting
- **Comprehensive evaluation**: Includes ROC-AUC, PR-AUC, confusion matrix, and cost analysis

## 📁 Project Structure

```
HealthCare-Provider-Fraud-Detection-Project/
│
├── Datasets/                          # Raw data files
│   ├── Train_Beneficiarydata-1542865627584.csv
│   ├── Train_Inpatientdata-1542865627584.csv
│   ├── Train_Outpatientdata-1542865627584.csv
│   ├── Train-1542865627584.csv
│   ├── Test_Beneficiarydata-1542969243754.csv
│   ├── Test_Inpatientdata-1542969243754.csv
│   ├── Test_Outpatientdata-1542969243754.csv
│   └── Test-1542969243754.csv
│
├── output/                            # Generated outputs
│   ├── processed_provider_data.csv    # Processed features
│   ├── best_model.pkl                 # Trained model
│   └── test_predictions.csv           # Model predictions
│
├── 01_data_exploration_and_feature_engineering.ipynb
├── modeling.ipynb
├── 03_evaluation.ipynb
└── README.md
```

## 🎯 Results Summary

### Model Performance

The **Random Forest Classifier** (after hyperparameter tuning) achieved the best performance:

| Metric | Score |
|--------|-------|
| **ROC-AUC** | 0.9554 |
| **PR-AUC** | 0.7570 |
| **Precision** | ~0.75+ |
| **Recall** | ~0.70+ |
| **F1-Score** | ~0.72+ |

### Model Comparison

| Model | ROC-AUC | PR-AUC |
|-------|---------|--------|
| Logistic Regression | 0.9554 | 0.7570 |
| Random Forest | 0.9429 | 0.6962 |
| Gradient Boosting | 0.9410 | 0.7290 |

### Key Insights

- **Class Imbalance**: The dataset has significant class imbalance with fraud cases being the minority
- **Financial Impact**: The model helps reduce costs by catching fraudulent providers while minimizing false alarms
- **Important Features**: Total reimbursed amount, claim volume, unique beneficiaries, and chronic conditions are strong predictors
- **Cost Analysis**: Based on estimated costs:
  - False Negative (missed fraud): $10,000 per case
  - False Positive (unnecessary investigation): $1,000 per case

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
```

### Required Libraries

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
joblib
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/H-1000/HealthCare-Provider-Fraud-Detection-Project.git
   cd HealthCare-Provider-Fraud-Detection-Project
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib
   ```

3. **Verify data files**
   Ensure all CSV files are present in the `Datasets/` folder.

## 📝 Reproduction Instructions

Follow these steps to reproduce the analysis:

### Step 1: Data Exploration and Feature Engineering

Open and run `01_data_exploration_and_feature_engineering.ipynb`

This notebook will:
- Load and merge beneficiary, inpatient, and outpatient data
- Perform data quality checks
- Engineer provider-level features (claim counts, financials, patient demographics)
- Conduct exploratory data analysis
- Save processed data to `output/processed_provider_data.csv`

**Key Outputs:**
- Processed provider-level dataset
- EDA visualizations (class distribution, correlations, geographic analysis)

### Step 2: Model Training

Open and run `modeling.ipynb`

This notebook will:
- Load the processed data
- Split data into training (80%) and test (20%) sets
- Create preprocessing pipeline (imputation, scaling, encoding)
- Train and compare multiple models (Logistic Regression, Random Forest, Gradient Boosting)
- Apply SMOTE for class imbalance handling
- Perform hyperparameter tuning on Random Forest
- Save the best model and predictions to `output/`

**Key Outputs:**
- `output/best_model.pkl` - Trained model
- `output/test_predictions.csv` - Test set predictions

### Step 3: Model Evaluation

Open and run `03_evaluation.ipynb`

This notebook will:
- Load test predictions
- Calculate comprehensive metrics (Precision, Recall, F1, ROC-AUC, PR-AUC)
- Generate confusion matrix
- Plot ROC and Precision-Recall curves
- Perform cost-based analysis
- Analyze error cases (false positives/negatives)
- Generate final evaluation report

**Key Outputs:**
- Performance metrics dashboard
- Evaluation visualizations
- Cost analysis report

## 💡 Usage

To use the trained model for predictions:

```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('output/best_model.pkl')

# Load your new data (must have same features as training data)
new_data = pd.read_csv('your_new_provider_data.csv')

# Make predictions
predictions = model.predict(new_data)
probabilities = model.predict_proba(new_data)[:, 1]

# Results
fraud_providers = new_data[predictions == 1]
print(f"Detected {len(fraud_providers)} potentially fraudulent providers")
```

## 📈 Future Improvements

- **Feature Enhancement**: Include more temporal patterns (seasonality, trends)
- **Model Ensemble**: Combine multiple models for better performance
- **Real-time Detection**: Develop streaming fraud detection system
- **Explainability**: Add SHAP or LIME for model interpretability
- **Deep Learning**: Experiment with neural networks for complex pattern detection
- **External Data**: Incorporate provider network data and external databases

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**H-1000**
- GitHub: [@H-1000](https://github.com/H-1000)

## 🙏 Acknowledgments

- Dataset source: Medicare claims data
- Inspiration: Healthcare fraud detection research
- Tools: scikit-learn, imbalanced-learn, pandas

---

**Note**: This project is for educational and research purposes. Always consult with domain experts and follow proper procedures when implementing fraud detection systems in production environments.
