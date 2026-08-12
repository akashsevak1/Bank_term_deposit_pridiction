# Bank Term Deposit Prediction

A machine learning project that predicts whether a bank customer is likely to subscribe to a term deposit based on customer demographics, financial information, and campaign-related attributes.

## 📌 Project Overview

The goal of this project is to build a classification model that helps banks identify customers who are more likely to subscribe to a term deposit.

The project uses the **UCI Bank Marketing dataset structure** and follows a complete machine learning workflow:

* Data loading
* Data exploration
* Data quality checks
* Exploratory Data Analysis (EDA)
* Feature preprocessing
* Model training
* Model comparison
* Cross-validation
* Model evaluation
* Detection of data leakage
* Final model selection

> **Dataset Note:** The notebook currently contains outputs generated using a small sample dataset with the same column structure as the original UCI Bank Marketing dataset. The notebook is designed to work with `train.csv` and `test.csv`.

---

## 🎯 Business Problem

Banks regularly contact customers to promote financial products such as term deposits.

However, contacting every customer can be inefficient.

This project aims to answer:

> **"Which customers are more likely to subscribe to a bank term deposit?"**

A predictive model can help the bank prioritize customers who have a higher probability of subscribing, potentially improving campaign efficiency.

---

## 📊 Dataset

The dataset contains customer and marketing campaign information.

The target variable is:

* `y` — whether the customer subscribed to a term deposit (`yes` / `no`)

Important features include:

| Feature     | Description                            |
| ----------- | -------------------------------------- |
| `age`       | Customer age                           |
| `job`       | Customer occupation                    |
| `marital`   | Marital status                         |
| `education` | Education level                        |
| `default`   | Credit default status                  |
| `balance`   | Average yearly balance                 |
| `housing`   | Housing loan status                    |
| `loan`      | Personal loan status                   |
| `contact`   | Contact communication type             |
| `day`       | Last contact day                       |
| `month`     | Last contact month                     |
| `duration`  | Last contact duration                  |
| `campaign`  | Number of contacts during the campaign |
| `pdays`     | Days since previous campaign contact   |
| `previous`  | Number of previous contacts            |
| `poutcome`  | Previous campaign outcome              |
| `y`         | Term deposit subscription target       |

These column meanings are documented directly in the notebook.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* Logistic Regression
* Random Forest
* One-Hot Encoding
* StandardScaler
* Cross-Validation
* ROC-AUC
* Classification Report
* Confusion Matrix

The notebook imports and uses these libraries for preprocessing, modeling, visualization, and evaluation.

---

## 🔎 Exploratory Data Analysis

The project performs initial data inspection including:

* Dataset shape
* Data types
* Statistical summary
* Missing-value analysis
* Duplicate-row detection
* Distribution and relationship analysis

## For the sample run, the training dataset contains **4,000 rows and 17 columns**, with no missing values and no duplicate rows detected.

## ⚙️ Data Preprocessing

The project separates the target variable `y` from the input features.

Categorical features are processed using:

```python
OneHotEncoder(handle_unknown='ignore')
```

Numerical features are standardized using:

```python
StandardScaler()
```

A `ColumnTransformer` and Scikit-learn `Pipeline` are used to combine preprocessing and model training.

The training data is further split into:

* Training set: **3,200 rows**
* Validation set: **800 rows**
* Test set: **800 rows**

The split uses stratification to account for class imbalance.

---

## 🤖 Machine Learning Models

Two classification algorithms were compared:

### 1. Logistic Regression

Used as an interpretable baseline classification model.

```python
LogisticRegression(
    class_weight='balanced',
    max_iter=1000
)
```

### 2. Random Forest

Used as a tree-based ensemble model.

```python
RandomForestClassifier(
    class_weight='balanced',
    n_estimators=300,
    random_state=42
)
```

## Class weighting was used to give additional importance to the minority `yes` class.

## ⚠️ Data Leakage Analysis

One of the important parts of this project is identifying **data leakage** caused by the `duration` feature.

`duration` represents the duration of the last contact. Since this information is only known after the call has happened, using it for a model intended to predict whether a customer should be contacted would not be realistic.

Therefore, the project compares:

* Models **with `duration`**
* Models **without `duration`**

The validation results show a significant performance drop when `duration` is removed.

For example:

| Feature Set      | Model               | Accuracy | ROC-AUC |
| ---------------- | ------------------- | -------: | ------: |
| With duration    | Logistic Regression |   84.63% |   0.918 |
| With duration    | Random Forest       |   83.38% |   0.913 |
| Without duration | Logistic Regression |   64.38% |   0.677 |
| Without duration | Random Forest       |   61.88% |   0.662 |

This comparison demonstrates how a feature that would not be available before a customer interaction can artificially inflate model performance.

---

## 🔄 Cross-Validation

A **5-fold Stratified Cross-Validation** was performed using the realistic feature set without `duration`.

The Logistic Regression model achieved an average ROC-AUC of approximately:

**0.657**

The individual fold ROC-AUC scores were:

```text
0.694
0.658
0.634
0.645
0.654
```

---

## 🏆 Final Model

The final selected model is:

**Logistic Regression with `duration` removed**

Configuration:

```python
LogisticRegression(
    class_weight='balanced',
    max_iter=1000
)
```

This model was selected because it represents a more realistic **pre-call prediction scenario**, where the bank does not yet know the duration of the customer's future interaction.

---

## 📈 Final Test Results

On the test dataset, the final model achieved:

| Metric          |     Score |
| --------------- | --------: |
| Accuracy        | **62.5%** |
| ROC-AUC         | **0.687** |
| Precision — No  |      0.68 |
| Recall — No     |      0.64 |
| F1 — No         |      0.66 |
| Precision — Yes |      0.57 |
| Recall — Yes    |      0.60 |
| F1 — Yes        |      0.58 |

The model was evaluated using accuracy, ROC-AUC, precision, recall, F1-score, and a confusion matrix.

---

## 💡 Key Learnings

This project demonstrates several important machine learning concepts:

1. **Data preprocessing** for numerical and categorical variables.
2. **Exploratory Data Analysis** to understand the dataset.
3. **Class imbalance handling** using `class_weight='balanced'`.
4. **Model comparison** between Logistic Regression and Random Forest.
5. **Data leakage detection** using the `duration` feature.
6. **Cross-validation** for checking model robustness.
7. **ROC-AUC, precision, recall and F1-score** for classification evaluation.
8. Building an end-to-end machine learning pipeline using Scikit-learn.

---

## 📁 Project Structure

```text
Bank-Term-Deposit-Prediction/
│
├── Bank_Term_Deposit_Predictions.ipynb
├── train.csv
├── test.csv
├── README.md
└── requirements.txt
```

> Keep the CSV files in the project folder if you want other users to run the notebook without changing the file paths.

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Bank-Term-Deposit-Prediction.git
```

### 2. Navigate to the project

```bash
cd Bank-Term-Deposit-Prediction
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open

```text
Bank_Term_Deposit_Predictions.ipynb
```

Make sure `train.csv` and `test.csv` are available in the same directory.

---

## 📌 Future Improvements

Possible improvements include:

* Hyperparameter tuning
* Feature importance analysis
* Threshold optimization
* Handling class imbalance using additional techniques
* Trying XGBoost or other boosting algorithms
* Building a Streamlit prediction application
* Deploying the final model as an API
* Adding a business-focused dashboard

---

## 👨‍💻 Author

**Aakash Sevak**

Data Analyst | Business Analyst | Python | SQL | Power BI | Machine Learning

---

## ⭐ Project Highlights

**Business Problem → Data Cleaning → EDA → Feature Engineering → Machine Learning → Model Comparison → Data Leakage Detection → Cross-Validation → Final Evaluation**

This project demonstrates an end-to-end approach to solving a real-world banking classification problem using Python and machine learning.
