# 💳 Credit Card Fraud Detection — ML Classification & Deployment
### Machine Learning (Assignment - 2) | M.Tech (AIML) — BITS Pilani WILP

---

## 👤 Student Information

| Field | Details |
|---|---|
| **Student Name** | **PADAMATA ANURAG** |
| **Roll Number** | **2026ak05613** |
| **Email Address** | **2026ak05613@wilp.bits-pilani.ac.in** |
| **Course** | Machine Learning (M.Tech in Artificial Intelligence & Machine Learning) |
| **Academic Division** | Work Integrated Learning Programmes (WILP), BITS Pilani |
| **Execution Platform** | **BITS Virtual Lab** |

---

## 📌 a. Problem Statement

Credit card fraud represents one of the most critical challenges in financial technology, resulting in tens of billions of dollars in annual losses worldwide. This project focuses on **binary classification** to predict whether a financial transaction is **fraudulent (`is_fraud = 1`)** or **legitimate (`is_fraud = 0`)** based on historical transaction attributes, behavioral markers, and account velocity indicators. 

Detecting fraud reliably in real time enables banking institutions and payment gateways to immediately decline fraudulent authorizations or trigger multi-factor step-up verification, safeguarding consumer capital while minimizing false declines for genuine cardholders. The problem presents significant machine learning challenges, notably **severe class imbalance** (less than 2.1% fraud), high feature dimensionality, and the requirement for robust calibration across recall and precision.

---

## 📊 b. Dataset Description

- **Dataset Name**: Credit Card Fraud Detection 2026
- **Source Repository**: Public Kaggle ML Repository ([Credit Card Fraud Detection](https://www.kaggle.com/datasets))
- **Total Instances**: **100,000 transactions** (exceeds minimum requirement of 500)
- **Total Features**: **34 predictive features** (after removing direct target leaks and raw identifiers, exceeding minimum requirement of 12)
- **Target Variable**: `is_fraud`
  - `0`: Legitimate Transaction (**97,945 instances / 97.95%**)
  - `1`: Fraudulent Transaction (**2,055 instances / 2.05%**)
- **Key Feature Categories**:
  - *Transaction Attributes*: `transaction_amount`, `merchant_category`, `merchant_country`, `transaction_hour`, `transaction_day_of_week`.
  - *Card & Customer Characteristics*: `card_type`, `card_brand`, `card_present`, `customer_age`, `gender`, `annual_income`, `credit_score`.
  - *Account History & Velocity Indicators*: `std_transaction_amount_30d`, `avg_transaction_amount_30d`, `transactions_last_24h`, `transactions_last_7d`, `num_declined_last_30d`, `num_chargebacks_ever`.
  - *Digital Footprint*: `device_type`, `ip_address_country`, `distance_from_home_km`, `email_domain`.

---

## 🔗 c. Repository & Deployment Links

- **GitHub Repository**: `https://github.com/PADAMATA-ANURAG/Credit-Card-Fraud-Detection-ML` *(Replace with your pushed repo URL)*
- **Live Streamlit Web App**: `https://credit-card-fraud-detection-2026.streamlit.app/` *(Replace with your deployed app URL)*
- **BITS Virtual Lab Execution Proof**: Captured on BITS Virtual Lab environment (included in PDF submission report).

### Repository Structure:
```
Project/
│── app.py                    # Streamlit interactive frontend application
│── requirements.txt          # Python dependencies for cloud deployment
│── README.md                 # Complete documentation and analytical report
│── test_data.csv             # Bundled stratified test dataset (1,000 samples)
│── train_evaluate.py         # Full training pipeline and benchmarking script
│── models/                   # 5 Machine Learning Classifiers (Built from scratch)
│   ├── logistic_regression.py
│   ├── decision_tree.py
│   ├── knn.py
│   ├── naive_bayes.py
│   └── svm.py
│── utils/                    # 6 Evaluation metrics & Preprocessing from scratch
│   ├── metrics.py
│   └── preprocessing.py
└── model/                    # Pre-trained model artifacts (.joblib)
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    └── preprocessor.joblib
```

---

## 🤖 d. Models Used & Evaluation Benchmark

All **5 classification models** were implemented **from scratch using pure NumPy** without relying on predefined scikit-learn estimators. All 6 required evaluation metrics were likewise calculated from first principles:
- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
- **AUC Score**: Area under the empirical ROC curve using numerical trapezoidal integration
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall**: $\frac{TP}{TP + FN}$
- **F1 Score**: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
- **Matthews Correlation Coefficient (MCC)**: $\frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$

### 1. Model Comparison Table (Test Set Evaluation — 20,000 Instances)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.5588 | **0.8356** | 0.0408 | **0.9099** | 0.0781 | 0.1314 |
| **Decision Tree Classifier** | **0.9752** | 0.6471 | 0.2237 | 0.0827 | 0.1208 | 0.1253 |
| **K-Nearest Neighbor (KNN, k=7)** | 0.9792 | 0.6337 | **0.2500** | 0.0049 | 0.0095 | 0.0324 |
| **Naive Bayes (Gaussian)** ⭐ | 0.9066 | 0.8193 | 0.1004 | 0.4453 | **0.1638** | **0.1782** |
| **Support Vector Machine (Linear SGD)** | 0.6425 | 0.8247 | 0.0475 | 0.8613 | 0.0901 | 0.1465 |

---

### 2. Model Observations Table

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Achieves the highest AUC (0.8356) and exceptional fraud recall (90.99%) due to balanced sample-weight gradient penalization. However, setting the decision threshold to prioritize sensitivity results in elevated false positives on legitimate accounts, lowering accuracy (55.88%) and precision (4.08%). It serves as an effective high-recall first-stage screening filter. |
| **Decision Tree Classifier** | Achieves high overall accuracy (97.52%) and respectable precision (22.37%) by creating axis-aligned rectangular partitions on high-gain velocity features. However, because recursive Gini impurity minimization naturally favors majority-class purifications, the tree suffers from low recall (8.27%), missing substantial fraudulent transactions that do not form dense isolated clusters. |
| **K-Nearest Neighbor (KNN)** | Displays high nominal accuracy (97.92%) but suffers in an extreme 47:1 class imbalance setting. Because legitimate points heavily outnumber fraud cases in local $k=7$ Euclidean neighborhoods, the majority vote rarely triggers for fraud, yielding an extremely low recall (0.49%) and F1 (0.0095). Distance-based metrics are also affected by feature sparsity in 34 dimensions. |
| **Naive Bayes (Gaussian)** ⭐ | **Overall Winner for this dataset.** Yields the highest balanced F1 Score (0.1638) and highest MCC Score (0.1782) with strong AUC (0.8193) and 44.53% recall. Class-conditional Gaussian likelihoods naturally capture distinct statistical distributions in fraud markers (e.g., elevated chargebacks, rapid transaction frequency) without overfitting to majority prevalence. |
| **Support Vector Machine (Linear SGD)** | Trained using class-weighted Hinge Loss with Mini-Batch SGD and Platt scaling. Delivers strong AUC (0.8247) and 86.13% recall. The linear separating hyperplane detects significant fraudulent patterns, though like Logistic Regression, regularized margin violations lead to trade-offs in precision (4.75%). |
| **Overall Winner for your dataset?** | **Gaussian Naive Bayes** is the clear winner for credit card fraud detection on this dataset. It achieves the best balance between catching fraudulent transactions (Recall = 44.53%), maintaining operational accuracy (90.66%), and minimizing unnecessary customer disruptions, as evidenced by top ranks in both **F1 Score (0.1638)** and **MCC (0.1782)**. |

---

## 💻 e. How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/PADAMATA-ANURAG/Credit-Card-Fraud-Detection-ML.git
cd Credit-Card-Fraud-Detection-ML
```

### 2. Create and activate a virtual environment (optional but recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install required packages
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application
```bash
streamlit run app.py
```
The application will launch automatically in your default browser at `http://localhost:8501`.

---

## 🚀 f. Deployment on Streamlit Community Cloud

1. Push this repository to your GitHub account.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **"New App"**, choose repository `PADAMATA-ANURAG/Credit-Card-Fraud-Detection-ML`, branch `main`, and main file `app.py`.
4. Click **Deploy**. Streamlit will automatically read `requirements.txt` and launch your live app.
