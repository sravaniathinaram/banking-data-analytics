# BANKING SYSTEM DATA ANALYSIS

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load Dataset
df = pd.read_csv("bank_data.csv")

# Step 3: Basic Info
print(df.head())
print(df.info())
print(df.describe())

# Step 4: Data Cleaning
df = df.drop_duplicates()
df = df.dropna(subset=["Age", "Account_Balance", "Credit_Score"])  # Remove missing key data

# Handle outliers
df["Account_Balance"] = np.where(df["Account_Balance"] > df["Account_Balance"].quantile(0.99),
                                 df["Account_Balance"].quantile(0.99),
                                 df["Account_Balance"])

# Step 5: Exploratory Data Analysis
sns.histplot(df["Account_Balance"], bins=30, kde=True)
plt.title("Distribution of Account Balance")
plt.show()

sns.boxplot(x="Gender", y="Account_Balance", data=df)
plt.title("Account Balance by Gender")
plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Step 6: Feature Analysis
avg_balance = df.groupby("Branch_Code")["Account_Balance"].mean().sort_values(ascending=False)
print("Top Branches by Average Balance:")
print(avg_balance.head())

# Step 7: Predictive Modeling (Optional Example)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

X = df[["Age", "Account_Balance", "Transaction_Count", "Credit_Score", "Loan_Amount"]]
y = df["Is_Default"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))