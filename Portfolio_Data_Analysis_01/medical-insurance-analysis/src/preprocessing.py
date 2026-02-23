# ================================
# IMPORT LIBRARY
# ================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setting style
sns.set(style="whitegrid")


# ================================
# LOAD DATA
# ================================
df = pd.read_csv("../data/medical_insurance_2026_kaggle.csv")

print("===== HEAD DATA =====")
print(df.head())

print("\n===== INFO DATA =====")
print(df.info())

print("\n===== DESCRIBE DATA =====")
print(df.describe())


# ================================
# CHECK MISSING VALUES
# ================================
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())


# ================================
# DATA PREPROCESSING
# ================================

# Convert datetime jika kolom ada
if 'record_date' in df.columns:
    df['record_date'] = pd.to_datetime(df['record_date'])

# Encoding smoker jika masih kategorikal
if 'smoker' in df.columns:
    df['smoker_flag'] = df['smoker'].map({'yes': 1, 'no': 0})

# Drop missing value
df = df.dropna()

print("\n===== DATA AFTER CLEANING =====")
print(df.head())


# ================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ================================

# Distribution charges
plt.figure(figsize=(8,5))
sns.histplot(df['charges'], kde=True)
plt.title("Distribution of Insurance Charges")
plt.show()


# Smoker vs Charges
if 'smoker' in df.columns:
    plt.figure(figsize=(6,5))
    sns.boxplot(x='smoker', y='charges', data=df)
    plt.title("Charges by Smoker Status")
    plt.show()


# BMI vs Charges
plt.figure(figsize=(6,5))
sns.scatterplot(x='bmi', y='charges', data=df)
plt.title("BMI vs Charges")
plt.show()


# Age vs Charges
plt.figure(figsize=(6,5))
sns.scatterplot(x='age', y='charges', data=df)
plt.title("Age vs Charges")
plt.show()


# Region vs Charges
if 'region' in df.columns:
    plt.figure(figsize=(6,5))
    sns.boxplot(x='region', y='charges', data=df)
    plt.title("Charges by Region")
    plt.show()


# Risk Score vs Charges (jika ada)
if 'risk_score' in df.columns:
    plt.figure(figsize=(6,5))
    sns.scatterplot(x='risk_score', y='charges', data=df)
    plt.title("Risk Score vs Charges")
    plt.show()


# Insurance Tier Analysis (jika ada)
if 'insurance_tier' in df.columns:
    plt.figure(figsize=(8,5))
    sns.barplot(x='insurance_tier', y='charges', data=df)
    plt.title("Average Charges by Insurance Tier")
    plt.show()


# ================================
# CORRELATION MATRIX
# ================================
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


# ================================
# MACHINE LEARNING MODEL
# ================================
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Feature selection
features = ['age', 'bmi', 'children']

if 'smoker_flag' in df.columns:
    features.append('smoker_flag')

if 'risk_score' in df.columns:
    features.append('risk_score')

X = df[features]
y = df['charges']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)


# ================================
# MODEL EVALUATION
# ================================
r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

print("\n===== MODEL EVALUATION =====")
print("R2 Score :", r2)
print("MAE      :", mae)
print("RMSE     :", rmse)


# ================================
# ACTUAL VS PREDICTION PLOT
# ================================
plt.figure(figsize=(6,6))
plt.scatter(y_test, pred)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted")
plt.show()


# ================================
# FEATURE IMPORTANCE (COEFFICIENT)
# ================================
coeff_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print("\n===== FEATURE IMPORTANCE =====")
print(coeff_df.sort_values(by="Coefficient", ascending=False))