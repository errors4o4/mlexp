# Libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Preprocessing:
# Drop the rows where the workclass is ?
df = df[df['workclass'] != '?']
# Drop the rows where the occupation is ?
df = df[df['occupation'] != '?']

# Label Encoding:
label_encoder = preprocessing.LabelEncoder()
df['education'] = label_encoder.fit_transform(df['education'])
df['marital.status'] = label_encoder.fit_transform(df['marital.status'])
df['occupation'] = label_encoder.fit_transform(df['occupation'])
df['workclass'] = label_encoder.fit_transform(df['workclass'])
df['relationship'] = label_encoder.fit_transform(df['relationship'])
df['race'] = label_encoder.fit_transform(df['race'])
df['sex'] = label_encoder.fit_transform(df['sex'])
df['native.country'] = label_encoder.fit_transform(df['native.country'])
df['income'] = label_encoder.fit_transform(df['income'])

# Dependent and Independent Variables:
X = df.drop('income', axis=1)
y = df['income']

# Train-test Split:
x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.8)

# Model Initialisation:
gini_clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)

# Model Training:
gini_clf.fit(x_train, y_train)


# Model Prediction:
y_pred = gini_clf.predict(x_test)

# Model Evaluation:
accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
