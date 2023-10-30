# Libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Preprocessing:
# Null Value Check
# Duplicates Check

# Dependent and Independent Variables:
x = df.drop(columns='MEDV')
y = df['MEDV']

# Train-test Split:
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8)

# Model Initialisation:
model = LinearRegression()

# Model Training:
model.fit(x_train,y_train)

# Model Prediction:
y_pred = model.predict(x_test)

# Model Evaluation:
print('Mean Squared Error: ', mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error: ', np.sqrt(mean_squared_error(y_test,y_pred)))

# Plot a Scatter plot:
plt.scatter(y_test,y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()
