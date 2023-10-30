# Libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Preprocessing:
# Drop Categorical Columns:
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Fill Null Values by mean:
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Label Encoding:
label_encoder = preprocessing.LabelEncoder()
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])
df['Sex'] = label_encoder.fit_transform(df['Sex'])

# Dependent and Independent Variables:
X = df.drop('Survived', axis=1)
y = df['Survived']

# Train-test Split:
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8)

# Model Initialisation:
model = LogisticRegression()

# Model Training:
model.fit(x_train,y_train)

# Model Prediction:
y_pred = model.predict(x_test)

# Model Evaluation:
accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
