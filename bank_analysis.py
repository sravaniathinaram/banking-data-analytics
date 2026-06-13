import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

df = pd.read_csv("bank_data.csv")

print("First 5 Records")
print(df.head())
print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())


df.drop_duplicates(inplace=True)

df.fillna(df.median(numeric_only=True), inplace=True)

print("\nData Cleaned Successfully")


print("\nStatistical Summary")
print(df.describe())


plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df['Account_Balance'], bins=20, kde=True)
plt.title("Account Balance Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='Loan_Approved', data=df)
plt.title("Loan Approval Status")
plt.show()
plt.figure(figsize=(8,6))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap='coolwarm'
)
plt.title("Correlation Matrix")
plt.show()


features = [
    'Age',
    'Account_Balance',
    'Transaction_Amount'
]

X = df[features]
y = df['Loan_Approved']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy Score")
print(accuracy)

print("\nConfusion Matrix")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report")
print(classification_report(y_test, predictions))
new_customer = pd.DataFrame({
    'Age': [35],
    'Account_Balance': [18000],
    'Transaction_Amount': [5000]
})

prediction = model.predict(new_customer)

if prediction[0] == 1:
    print("\nLoan Approved")
else:
    print("\nLoan Not Approved")
