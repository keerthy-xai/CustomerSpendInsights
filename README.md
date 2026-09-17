# Customer Spending Score Prediction using Linear Regression

## 📌 Project Overview

This project implements a **Supervised Machine Learning** approach to predict a customer's **Spending Score (1-100)** based on their **Annual Income (k$)**.

The project uses the **Mall Customers Enhanced** dataset and demonstrates the complete basic machine learning workflow, including:

- Data loading and inspection
- Data shape and sample analysis
- Missing-value checking
- Duplicate-record checking
- Outlier detection
- Missing-value handling
- Categorical data encoding
- Feature scaling
- Feature selection
- Linear Regression implementation from scratch
- Prediction on test data
- Model evaluation using R² Score and Mean Squared Error
- Linear Regression using Scikit-learn
- Data visualization

## 🎯 Objective

The main objective is to understand how **Annual Income** is related to a customer's **Spending Score** and to implement Linear Regression using both:

1. A manual Gradient Descent approach without Scikit-learn
2. Scikit-learn's `LinearRegression` model

## 📊 Dataset

The project uses:

`Mall_Customers_Enhanced.csv`

The dataset contains **200 records and 10 columns**.

### Features

| Column | Description |
|---|---|
| CustomerID | Unique customer identifier |
| Gender | Customer gender |
| Age | Customer age |
| Annual Income (k$) | Annual income in thousands of dollars |
| Spending Score (1-100) | Customer spending score |
| Age Group | Customer age category |
| Estimated Savings (k$) | Estimated savings in thousands of dollars |
| Credit Score | Customer credit score |
| Loyalty Years | Number of loyalty years |
| Preferred Category | Preferred customer category |

### Target Variable

**Spending Score (1-100)**

### Independent Variable

**Annual Income (k$)**

## 🧠 Machine Learning Algorithm

### Linear Regression

Linear Regression is a supervised learning algorithm used to model the relationship between an independent variable and a continuous dependent variable.

In this project:

- **Input:** Annual Income (k$)
- **Output:** Spending Score (1-100)

The project first implements Linear Regression manually using **Gradient Descent** and then uses Scikit-learn's `LinearRegression`.

## 🔄 Project Workflow

```text
Load Dataset
     ↓
Explore Dataset
     ↓
Check Data Types
     ↓
Check Missing Values
     ↓
Check Duplicate Records
     ↓
Detect Outliers
     ↓
Handle Missing Values
     ↓
Encode Categorical Data
     ↓
Standardize Numerical Features
     ↓
Select Features and Target
     ↓
Implement Linear Regression
     ↓
Train/Test Split
     ↓
Make Predictions
     ↓
Evaluate Model
     ↓
Visualize Results
```

## 🧹 Data Preprocessing

### 1. Data Inspection

The dataset was loaded using Pandas and its data types were examined.

### 2. Missing Values

The dataset contains **4 missing values in the `Age Group` column**. The notebook applies mode-based handling for this column.

The notebook also includes mean-based handling for `Credit Score`.

### 3. Duplicate Records

No duplicate records were found in the dataset.

### 4. Outlier Detection

The project uses the **Interquartile Range (IQR)** method for basic outlier detection.

For `Annual Income (k$)`, **2 outliers** were detected.

### 5. Encoding

Categorical data was encoded using:

- `LabelEncoder`
- `pd.get_dummies()`

### 6. Feature Scaling

`StandardScaler` from Scikit-learn was used to standardize numerical columns.

## 🛠️ Technologies Used

- Python
- Anaconda
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## 📦 Python Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
```

## ⚙️ Linear Regression From Scratch

The notebook implements Linear Regression without using Scikit-learn's regression model.

The implementation uses:

- Initial intercept `b0 = 0`
- Initial slope `b1 = 0`
- Learning rate `alpha = 0.0001`
- `1000` epochs
- Mean Squared Error as the cost function
- Gradient Descent for parameter optimization

The final values reported in the notebook are:

- **Intercept (b0):** 1.5489598387
- **Slope (b1):** 0.6782643984
- **Final Cost:** 1027.6301582375

## 📈 Manual Model Evaluation

The manually implemented model produced:

**R² Score:** `-0.5487207595`

The notebook also generates predictions for the test portion of the data.

## 🤖 Linear Regression Using Scikit-learn

The project uses an 80/20 train-test split:

- Training samples: **160**
- Testing samples: **40**

The model is created using:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
```

### Evaluation Results

The notebook reports:

| Metric | Result |
|---|---:|
| R² Score | -0.0298379414 |
| Mean Squared Error | 671.8244358038 |

These results are included directly from the executed notebook.

## 📊 Visualizations

The project includes visualizations for:

- Outlier detection using boxplots
- Actual vs. predicted values
- Linear Regression visualization with the regression line

## 🚀 How to Run the Project

### 1. Install Anaconda

Install Anaconda and launch **Jupyter Notebook** or **JupyterLab**.

### 2. Open the Project

Open the provided `.ipynb` notebook in Jupyter.

### 3. Add the Dataset

Make sure the following file is available in the same working directory as the notebook:

```text
Mall_Customers_Enhanced.csv
```

### 4. Run the Notebook

Run the notebook cells from top to bottom.

## 📁 Project Structure

```text
Customer-Spending-Score-Linear-Regression/
│
├── Mall_Customers_Enhanced.csv
├── supervised_learning_project.ipynb
└── README.md
```

## 📌 Key Learning Outcomes

Through this project, the following concepts were practiced:

- Understanding a dataset using Pandas
- Identifying missing values and duplicate records
- Detecting outliers using IQR
- Encoding categorical variables
- Standardizing numerical features
- Understanding independent and dependent variables
- Implementing Linear Regression manually
- Understanding Gradient Descent
- Splitting data into training and testing sets
- Using Scikit-learn for machine learning
- Evaluating regression models using R² Score and MSE
- Visualizing actual and predicted values

## 🔮 Future Improvements

Possible improvements to the project include:

- Using multiple independent variables instead of only Annual Income
- Comparing Linear Regression with other supervised learning algorithms
- Performing more detailed feature engineering
- Improving model performance through feature selection
- Using cross-validation
- Tuning model parameters
- Deploying the trained model as a web application

## 👩‍💻 Author

**Keerthana**

B.Tech Graduate  
Department of Artificial Intelligence and Data Science

---

⭐ This project was developed as a supervised machine learning practice project using Python, Anaconda, and Jupyter Notebook.
