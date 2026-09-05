# Pima Indians Diabetes Prediction

## Project Overview

This project uses the Pima Indians diabetes dataset to build a machine learning classification model for predicting the `Outcome` variable.

The notebook covers:

* Data loading and inspection
* Descriptive statistics
* Missing value checks
* Correlation analysis
* Exploratory data visualization
* Train-test splitting
* Logistic Regression model training
* Classification evaluation using a confusion matrix and classification report

---

## Dataset

The project uses the following dataset:

```text
diabetes.csv
```

### Dataset Size

* **Rows:** 768
* **Columns:** 9

### Features

The dataset contains the following columns:

| Feature                  | Data Type |
| ------------------------ | --------- |
| Pregnancies              | int64     |
| Glucose                  | int64     |
| BloodPressure            | int64     |
| SkinThickness            | int64     |
| Insulin                  | int64     |
| BMI                      | float64   |
| DiabetesPedigreeFunction | float64   |
| Age                      | int64     |
| Outcome                  | int64     |

The `Outcome` column is used as the target variable.

---

## Technologies and Libraries

The notebook uses Python with the following libraries:

* **NumPy**
* **Pandas**
* **Seaborn**
* **Matplotlib**
* **Scikit-learn**

---

## Data Loading

The dataset is loaded using Pandas:

```python
df = pd.read_csv('diabetes.csv')
```

The notebook then examines the dataset using:

```python
df.head()
df.describe()
df.info()
```

---

## Missing Values

Missing values were checked using:

```python
df.isnull().sum()
```

The notebook output shows **0 null values** in all columns.

> Note: The notebook checks for null values using `isnull()`. No additional treatment of zero values is performed in the notebook.

---

## Exploratory Data Analysis

### Correlation Analysis

A correlation matrix was calculated:

```python
a = df.corr()
```

The correlations were visualized using a heatmap:

```python
sns.heatmap(a, cmap='rocket', annot=True, linewidth=2)
```

---

## Data Visualizations

The notebook includes several visualizations using Seaborn.

### Age Distribution

```python
sns.histplot(df['Age'], kde=True)
```

### Outcome vs Glucose

```python
sns.barplot(x='Outcome', y='Glucose', data=df)
```

### Outcome vs Blood Pressure

```python
sns.barplot(x='Outcome', y='BloodPressure', data=df)
```

### Outcome vs Diabetes Pedigree Function

```python
sns.barplot(
    x='Outcome',
    y='DiabetesPedigreeFunction',
    data=df
)
```

### Outcome vs Pregnancies

```python
sns.barplot(x='Outcome', y='Pregnancies', data=df)
```

### Outcome vs BMI

```python
sns.barplot(x='Outcome', y='BMI', data=df)
```

### Outcome vs Age

```python
sns.barplot(x='Outcome', y='Age', data=df)
```

The notebook also sets the Seaborn plotting style to:

```python
sns.set_style('darkgrid')
```

---

## Feature and Target Selection

The dataset is divided into features (`X`) and the target (`y`).

```python
X = df.drop('Outcome', axis=1)
y = df['Outcome']
```

The features used for training are:

* Pregnancies
* Glucose
* BloodPressure
* SkinThickness
* Insulin
* BMI
* DiabetesPedigreeFunction
* Age

The target variable is:

```text
Outcome
```

---

## Train-Test Split

The dataset is split into training and testing sets using a **70/30 split**:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=101
)
```

This produced a test set containing **231 samples**, as shown in the classification report.

---

## Machine Learning Model

The project uses **Logistic Regression** from Scikit-learn.

```python
from sklearn.linear_model import LogisticRegression
```

The model is configured as:

```python
lr = LogisticRegression(
    solver='lbfgs',
    max_iter=1000
)
```

The model is trained using:

```python
lr.fit(X_train, y_train)
```

Predictions are generated with:

```python
predictions = lr.predict(X_test)
```

---

## Model Evaluation

The model is evaluated using:

* Confusion Matrix
* Precision
* Recall
* F1-score
* Accuracy

### Confusion Matrix

```text
[[132  18]
 [ 32  49]]
```

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
| ----- | --------: | -----: | -------: | ------: |
| 0     |      0.80 |   0.88 |     0.84 |     150 |
| 1     |      0.73 |   0.60 |     0.66 |      81 |

### Overall Results

| Metric                     | Score |
| -------------------------- | ----: |
| Accuracy                   |  0.78 |
| Macro Average Precision    |  0.77 |
| Macro Average Recall       |  0.74 |
| Macro Average F1-Score     |  0.75 |
| Weighted Average Precision |  0.78 |
| Weighted Average Recall    |  0.78 |
| Weighted Average F1-Score  |  0.78 |

---

## Project Structure

```text
├── Pima Indians.ipynb
├── diabetes.csv
└── README.md
```

---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the Project Directory

```bash
cd <your-repository-name>
```

### 3. Install the Required Libraries

```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open the Notebook

Open:

```text
Pima Indians.ipynb
```

Ensure that:

```text
diabetes.csv
```

is available in the working directory.

---

## Summary

This project applies Logistic Regression to the Pima Indians diabetes dataset using eight input features to predict the `Outcome` variable.

The notebook performs exploratory analysis, correlation visualization, train-test splitting, Logistic Regression training, and classification evaluation.

The recorded model accuracy on the test set is **0.78**.

---


