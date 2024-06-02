import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

path = 'WineQT.csv'  # Path input for dataframe

df = pd.read_csv(path)  # Reads the path
print(df.isnull().sum())

for col in df.columns():  # iterates through each column
    if df[col].isnull().sum() > 0:  # true if there are empty entries
        df[col] = df[col].fillna(df[col].mean())  # fills these empty entries with the mean
