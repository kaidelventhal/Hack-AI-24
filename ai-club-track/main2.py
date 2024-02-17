import numpy as np
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("HousingDataset.csv")

fit = numpy.array()
score = numpy.array()
headers = ["price", "area", "bedrooms", "bathrooms", "stories", "mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "parking", "prefarea", "furnishingstatus"]

i = 1
for i in range(14): 
  df_binary = df[['price', headers[i]]]
  df_binary.columns = ['price', headers[i]]
  df_binary.head()

  X = np.array(df_binary['price']).reshape(-1, 1)
  y = np.array(df_binary[headers[i]]).reshape(-1, 1)

  df_binary.dropna(inplace = True)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

  regr = LinearRegression()

  fit = regr.fit(X_train, y_train)
  score[i-1] = regr.score(X_test, y_test)

print("This is the linreg fits")
print(fit)
print("This is the linreg scores")