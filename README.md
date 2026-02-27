# EXNO2DS
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
    

# ----------------------------------------
# Step 1: Import Required Packages
# ----------------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------------------
# Step 2: Load the Dataset
# ----------------------------------------
data = pd.read_csv("Exp_2_dataset_titanic_dataset.csv")

print("\nDataset Loaded Successfully\n")
print(data.head())
print("\nDataset Info:\n")
print(data.info())
print(data.describe())
# ----------------------------------------
# Step 3: Data Cleansing - Handle Missing Values
# ----------------------------------------
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = data[column].fillna(data[column].mode()[0])   # Mode for categorical
    else:
        data[column] = data[column].fillna(data[column].median())   # Median for numerical

print("\nMissing values handled successfully.\n")

# ----------------------------------------
# Step 4: Boxplot to Analyze Outliers (Age & Fare)
# ----------------------------------------
plt.figure(figsize=(6,4))
sns.boxplot(x=data["Age"])
plt.title("Boxplot - Age")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x=data["Fare"])
plt.title("Boxplot - Fare")
plt.show()

# ----------------------------------------
# Step 5: Remove Outliers Using IQR Method
# ----------------------------------------
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

data = remove_outliers_iqr(data, "Age")
data = remove_outliers_iqr(data, "Fare")

print("Outliers removed using IQR method.\n")

# ----------------------------------------
# Step 6: Countplot for Categorical Data
# ----------------------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="Survived", data=data)
plt.title("Countplot - Survival Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="Sex", data=data)
plt.title("Countplot - Gender Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="Pclass", data=data)
plt.title("Countplot - Passenger Class Distribution")
plt.show()

# ----------------------------------------
# Step 7: Displot for Univariate Distribution
# ----------------------------------------
sns.displot(data["Age"], kde=True, height=4, aspect=1.5)
plt.title("Displot - Age Distribution")
plt.show()

sns.displot(data["Fare"], kde=True, height=4, aspect=1.5)
plt.title("Displot - Fare Distribution")
plt.show()

# ----------------------------------------
# Step 8: Cross Tabulation
# ----------------------------------------
print("\nCross Tabulation: Sex vs Survived\n")
print(pd.crosstab(data["Sex"], data["Survived"]))

print("\nCross Tabulation: Pclass vs Survived\n")
print(pd.crosstab(data["Pclass"], data["Survived"]))

# ----------------------------------------
# Step 9: Heatmap for Correlation Analysis
# ----------------------------------------
plt.figure(figsize=(8,6))
correlation_matrix = data.select_dtypes(include=np.number).corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap - Titanic Dataset")
plt.show()

<img width="1198" height="771" alt="image" src="https://github.com/user-attachments/assets/6431950f-ac46-42c0-9dde-da8039fd0ffb" />

<img width="1163" height="538" alt="image" src="https://github.com/user-attachments/assets/5d51d9d2-cbf7-49df-a63b-df6b2e6a3c25" />

<img width="837" height="568" alt="image" src="https://github.com/user-attachments/assets/a621961f-b11e-4d63-b6a5-4e4ad693d11b" />

<img width="974" height="598" alt="image" src="https://github.com/user-attachments/assets/99783075-c57f-4981-8516-24dd75d3f60c" />

<img width="1000" height="547" alt="image" src="https://github.com/user-attachments/assets/209e2644-864e-4d03-93fa-bada0c5a0c4a" />

<img width="839" height="544" alt="image" src="https://github.com/user-attachments/assets/bb64a676-dbb3-4658-ac88-cca35f25d3ae" />

<img width="950" height="594" alt="image" src="https://github.com/user-attachments/assets/d8ad3fdc-4087-4173-94d6-c2a2b636c405" />

<img width="1003" height="613" alt="image" src="https://github.com/user-attachments/assets/e9395f07-0a6d-45b6-8ae5-29aa53931124" />

<img width="603" height="388" alt="image" src="https://github.com/user-attachments/assets/733fc599-cc92-4cf0-a846-255c0009b302" />

<img width="957" height="806" alt="image" src="https://github.com/user-attachments/assets/31a7b1f3-7f5c-4922-95de-35094c8400ee" />

# RESULT
 Hence, Exploratory Data Analysis is performed  on the given data set.
        
