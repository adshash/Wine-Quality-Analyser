import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


def histogram():
    df.hist(bins=20, figsize=(10, 6))
    plt.show()

def barchart(xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    plt.bar(df[xlabel], df[ylabel])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{xlabel} against {ylabel}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


warnings.filterwarnings('ignore')

path = 'WineQT.csv'  # Path input for dataframe

df = pd.read_csv(path)  # Reads the path

for col in df.columns:  # iterates through each column
    if df[col].isnull().sum() > 0:  # true if there are empty entries
        df[col] = df[col].fillna(df[col].mean())  # fills these empty entries with the mean

print(df.describe().T)  # takes the transposition of the description of the df
barchart('quality', 'alcohol')  # shows how little variance there is of alcohol, relative to the quality

plt.figure(figsize=(12, 6))
sb.heatmap(df.corr() > 0.7, annot=True, cbar=False)  # shows a heatmap, displaying values where there is correlation
plt.show()
# As there is no correlation between any 2 columns, there isn't a need to ommit any two columns.


modesOfQuality = df['quality'].mode()  # finds the most common quality of wine

if len(modesOfQuality) > 1:  # checks to see if there are several modes of quality
    modeQuality = modesOfQuality.iloc[1]  # selects the 2nd one if there are multiple
else:
    modeQuality = modesOfQuality.iloc[0]
df['best quality'] = [1 if x > modeQuality else 0 for x in df.quality]  # if quality > modal quality => best wine

features = df.drop(['quality', 'best quality'], axis=1)  # creates df of feature data
target = df['best quality']  # same for target data

# Separates data into an 80:20 split for training and validation
xtrain, xtest, ytrain, ytest = train_test_split(features, target, test_size=0.2, random_state=40)

print(xtrain.shape, xtest.shape) # prints dimensions of the data arrays for the test and validation data
