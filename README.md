# BITS Pilani — WILP | M.Tech (AIML)
## Machine Learning — Assignment 2: Classification Models & Cloud Deployment
### Student: PADAMATA ANURAG | Roll Number: 2026ak05613 | Email: 2026ak05613@wilp.bits-pilani.ac.in

---

## 👤 Student Information & Submission Details

| Field | Submission Details |
|---|---|
| **Student Name** | **PADAMATA ANURAG** |
| **Roll Number** | **2026ak05613** |
| **Email Address** | **2026ak05613@wilp.bits-pilani.ac.in** |
| **Course** | Machine Learning (M.Tech AIML) |
| **Academic Division** | Work Integrated Learning Programmes (WILP), BITS Pilani |
| **Execution Platform** | **BITS Virtual Lab** |
| **Assignment** | Machine Learning Assignment - 2 (Total: 15 Marks) |

---

## a. Problem Statement

Financial institutions and electronic payment networks process billions of credit card transactions daily. Among these transactions, fraudulent operations generate severe financial damages and compromise consumer trust. This project addresses the **binary classification problem** of predicting whether a transaction is **fraudulent (`is_fraud = 1`)** or **legitimate (`is_fraud = 0`)** using customer demographics, transaction metadata, card behavioral indicators, and velocity statistics.

The practical goal is to deploy an automated classification pipeline capable of flagging fraudulent authorizations in real time with high sensitivity, while simultaneously minimizing false positives to avoid needlessly declining legitimate customer transactions. The primary machine learning challenges include **extreme class imbalance** (less than 2.1% fraud cases in 100,000 transactions), high feature dimensionality (34 continuous and categorical features), and multi-collinear spending velocity attributes.

---

## b. Dataset Description [ 1 Mark ]

- **Dataset Name**: Credit Card Fraud Detection Dataset 2026
- **Dataset Author / Publisher**: Mohan Krishnathalla
- **Public Source**: Kaggle Public Repository ([Credit Card Fraud Detection Dataset 2026](https://www.kaggle.com/datasets/mohankrishnathalla/credit-card-fraud-detection-dataset-2026))
- **Dataset URL**: `https://www.kaggle.com/datasets/mohankrishnathalla/credit-card-fraud-detection-dataset-2026`
- **Total Instance Size**: **100,000 instances / rows** (Substantially exceeds the minimum requirement of 500 instances)
- **Total Feature Size**: **34 predictive features** after cleaning (Substantially exceeds the minimum requirement of 12 features)
- **Target Variable**: `is_fraud` (Binary label: `0` for Legitimate, `1` for Fraudulent)
- **Class Distribution**:
  - `0` (Legitimate): **97,945 transactions (97.95%)**
  - `1` (Fraudulent): **2,055 transactions (2.05%)**
  - Imbalance Ratio: Approximately **47.7 : 1**
- **Key Feature Categories**:
  - *Monetary & Temporal Features*: `transaction_amount`, `transaction_hour`, `transaction_day_of_week`, `merchant_category`, `merchant_country`.
  - *Card & Customer Profile*: `card_type`, `card_brand`, `card_present`, `customer_age`, `gender`, `annual_income`, `credit_score`.
  - *Account History & Velocity*: `std_transaction_amount_30d`, `avg_transaction_amount_30d`, `transactions_last_24h`, `transactions_last_7d`, `num_declined_last_30d`, `num_chargebacks_ever`.
  - *Digital & Geographic Indicators*: `device_type`, `ip_address_country`, `distance_from_home_km`, `email_domain`.
- **Data Preprocessing**:
  - Direct target leaks (`fraud_type`, `fraud_amount_lost`) and non-predictive primary identifiers (`transaction_id`, `customer_id`, `card_id`, `merchant_id`, `transaction_datetime`) were removed.
  - Categorical variables were mapped via `LabelEncoderScratch`.
  - Continuous numerical features were normalized using `StandardScalerScratch` ($z = \frac{x - \mu}{\sigma}$).

---

## c. GitHub Repository Link [ 1 Mark ]

- **GitHub Repository URL**: [https://github.com/PADAMATA-ANURAG/Credit-Card-Fraud-Detection-ML](https://github.com/PADAMATA-ANURAG/Credit-Card-Fraud-Detection-ML)
- **Live Streamlit App URL**: [https://credit-card-fraud-detection-2026.streamlit.app/](https://credit-card-fraud-detection-2026.streamlit.app/)

### Complete Repository Structure:
```
project-folder/
├── app.py                    <- Interactive Streamlit web application
├── requirements.txt          <- Required Python package dependencies
├── README.md                 <- Step 5 comprehensive report (also in submission PDF)
├── test_data.csv             <- Bundled stratified test data (20,000 rows, 35 columns)
├── train_evaluate.py         <- End-to-end Python script training all 5 models
├── notebook.ipynb            <- Jupyter Notebook with complete EDA and evaluation
├── models/                   <- All 5 ML algorithms implemented from scratch (NumPy)
│   ├── logistic_regression.py
│   ├── decision_tree.py
│   ├── knn.py
│   ├── naive_bayes.py
│   └── svm.py
├── utils/                    <- 6 Evaluation metrics & Preprocessing from scratch
│   ├── metrics.py
│   └── preprocessing.py
└── model/                    <- Pre-trained serialised model artifacts (.joblib)
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    └── preprocessor.joblib
```

---

## d. Models Used: [ 5 Marks — 1 Mark per Model ]

All 5 classification models and all 6 evaluation metrics were built **from scratch using pure NumPy** without relying on predefined scikit-learn estimators. The parametric models (Logistic Regression, Decision Tree, Naive Bayes, Linear SVM) were trained on the complete 80,000-sample training partition. K-Nearest Neighbor (KNN) utilizes a stratified 10,000-sample training subset to ensure sub-second inference latency and comply with Streamlit Community Cloud resource bounds (1GB RAM). All 5 models are evaluated on the identical 20,000-sample stratified test partition.

### 1. Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | 0.5587 | **0.8356** | 0.0408 | **0.9100** | 0.0781 | 0.1314 |
| **Decision Tree** | 0.9751 | 0.6501 | 0.2179 | 0.0827 | 0.1199 | 0.1234 |
| **KNN (k=7)** | **0.9792** | 0.6201 | 0.1667 | 0.0024 | 0.0048 | 0.0178 |
| **Naive Bayes (Gaussian)** | 0.9066 | 0.8193 | 0.1004 | 0.4453 | **0.1638** | **0.1782** |
| **Support Vector Machine (Linear SGD)** | 0.6675 | 0.8255 | 0.0499 | 0.8418 | 0.0942 | 0.1507 |

---

### 2. Model Observations Table [ 3 Marks ]

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Demonstrates the highest discrimination capability across probability thresholds with an **AUC of 0.8356** and exceptional **Recall of 91.00%** (detecting 374 out of 411 fraud cases in the test set). Because the model utilizes class-weighted gradient updates to counteract the 47:1 class imbalance, its decision boundary is shifted to catch rare frauds, which causes a trade-off in nominal Accuracy (55.87%) and Precision (4.08%) due to higher false positives. It serves as an excellent high-sensitivity early screening stage. |
| **Decision Tree** | Achieves a high overall **Accuracy of 97.51%** and an **AUC of 0.6501**, with **Precision of 21.79%** and **Recall of 8.27%** (F1 Score of 0.1199, MCC of 0.1234). By recursively evaluating Gini impurity across percentile-sampled feature splits (such as transaction amount and velocity indicators), the CART tree partitions decision boundaries into orthogonal hyperplanes. In a 47:1 imbalanced setting, majority-class leaf nodes dominate unless deeply split, keeping minority recall modest while preserving high overall accuracy. |
| **KNN** | Yields high overall **Accuracy (97.92%)** and a **Precision of 16.67%**, but suffers severely from the curse of dimensionality across 34 normalized features combined with extreme class imbalance. In a $k=7$ nearest-neighbor sphere, legitimate transactions statistically dominate local Euclidean space, preventing the minority vote from surpassing the threshold. Consequently, the model achieves a low **Recall of 0.24%** and **F1 Score of 0.0048**, rendering standard Euclidean KNN unsuitable for raw imbalanced fraud detection. |
| **Naive Bayes** | Displays the most robust balanced performance across all dimensions, achieving the highest **F1 Score (0.1638)** and the highest **MCC Score (0.1782)** while maintaining **90.66% Accuracy**, **0.8193 AUC**, and **44.53% Recall** (detecting 183 fraud attacks). By modeling class-conditional feature distributions independently via Gaussian probability density functions, it captures strong abnormal shifts in fraud indicators without becoming biased toward the majority class prior. |
| **Support Vector Machine** | Implemented using class-weighted Hinge Loss with Mini-Batch Stochastic Gradient Descent and sigmoid-transformed margin probability approximation. The linear hyperplane establishes effective global separation, reaching a strong **AUC of 0.8255** and high **Recall of 84.18%** (identifying 346 fraud transactions) with an **Accuracy of 66.75%** and **MCC of 0.1507**. Like Logistic Regression, the margin penalties for minority misses drive a trade-off against precision (4.99%). |
| **Overall, Winner for your dataset?** | **Gaussian Naive Bayes is the overall winner for this Credit Card Fraud dataset.** In high-stakes fraud detection with severe class imbalance (97.95% vs 2.05%), raw Accuracy is deceptive. Naive Bayes delivers the best operational trade-off: it catches **44.53% of all fraud attacks** while maintaining a high **90.66% overall accuracy** and the highest **F1 Score (0.1638)** and **MCC Score (0.1782)** among all 5 models. |

---

## e. Instructions to Run Locally & Deploy

### Running the App Locally:
```bash
# 1. Navigate to the project directory
cd "C:\Users\DELL\Desktop\Assignment - 2 ML\Project"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Streamlit web app
streamlit run app.py
```

### Deploying to Streamlit Community Cloud:
1. Fork or push the `Project/` folder contents to your GitHub repository.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/) with your GitHub account.
3. Select your repository `Credit-Card-Fraud-Detection-ML`, branch `main`, and main file path `app.py`.
4. Click **Deploy** — the app will build and run live.
