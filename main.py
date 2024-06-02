import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import warnings
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


# Function to predict the classification of a new wine sample
def predict_wine_quality(wine_features, df):
    # Ensure the input is in the correct format (e.g., a DataFrame or a list of feature values)
    if isinstance(wine_features, pd.DataFrame):  # checks the type of data inputed
        new_data = wine_features
    else:
        new_data = pd.DataFrame([wine_features], columns=features.columns)  # creates a pd dataframe
    for col in new_data.columns:  # for all the columns, if there are empty entries, it will fill them with mean of
        # input data
        if new_data[col].isnull().sum() > 0:
            new_data[col] = new_data[col].fillna(df[col].mean())

    new_data_scaled = norm.transform(new_data)  # normalise input data

    # Predict the classification
    prediction = BestModel.predict(new_data_scaled)
    return 'Best Quality' if prediction[0] == 1 else 'Not Best Quality'


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

print(xtrain.shape, xtest.shape)  # prints dimensions of the data arrays for the test and validation data

norm = MinMaxScaler()  # Uses min max scaling from scikit to transform data to relative scale between 0 and 1
xtrain = norm.fit_transform(xtrain)  # fits to data then transforms it
xtest = norm.transform(xtest)

models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]

BestModel = models[0]  # Ensures BestModel is never undefined
correlation = np.inf  # initialises the correlation score

for i in range(3):
    models[i].fit(xtrain, ytrain)  # fits the training data using the model

    print(f'{models[i]}')
    TrainingAccuracy = metrics.roc_auc_score(ytrain, models[i].predict(xtrain))
    ValidationAccuracy = metrics.roc_auc_score(ytest, models[i].predict(xtest))
    print('Training Accuracy : ', TrainingAccuracy)  # produces a scalar value
    # signifying how accurate the model is. 1 = perfect, 0.5 = no better than random chance.
    print('Validation Accuracy : ', ValidationAccuracy)
    if abs(TrainingAccuracy - ValidationAccuracy) < correlation:
        correlation = abs(TrainingAccuracy - ValidationAccuracy)
        BestModel = models[i]
    print()

cm = metrics.confusion_matrix(ytest, BestModel.predict(xtest))  # creates confusion matrix of best model
disp = metrics.ConfusionMatrixDisplay(cm)  # displays the heatmap of the best model
disp.plot()
plt.show()

print(metrics.classification_report(ytest, BestModel.predict(xtest)))  # Prints the classification report

# Example usage: predict the quality of a new wine sample
new_wine = {
    'fixed acidity': 7.4,
    'volatile acidity': 0.7,
    'citric acid': 0.0,
    'residual sugar': 1.9,
    'chlorides': 0.076,
    'free sulfur dioxide': 11.0,
    'total sulfur dioxide': 34.0,
    'density': 0.9978,
    'pH': 3.51,
    'sulphates': 0.56,
    'alcohol': 9.4
}
print(predict_wine_quality(new_wine, df))