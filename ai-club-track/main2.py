import numpy as np
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv("HousingDataset.csv")
df_binary = df[['price', 'area']]
df_binary.columns = ['price', 'area']
df_binary.head()

X = np.array(df_binary['price']).reshape(-1, 1)
y = np.array(df_binary['area']).reshape(-1, 1)

df_binary.dropna(inplace = True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

regr = LinearRegression()

regr.fit(X_train, y_train)

print("This is the linreg score")
print(regr.score(X_test, y_test))