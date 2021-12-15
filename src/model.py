import numpy as np
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# open the data
data = pd.read_csv('../data/Housing.csv')

# get our X and y
X = data.iloc[:, 1:5].values
y = data.iloc[:, 0].values

#mix the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# train our model using a linearRegression and save the model
model_linear = LinearRegression().fit(X_train, y_train)
joblib.dump(model_linear, "../model/model_linear_regression.joblib")

# train our model using a logisticRegression and save the model
model_logistic = LogisticRegression().fit(X_train, y_train)
joblib.dump(model_logistic, "../model/model_logistic_regression.joblib")
