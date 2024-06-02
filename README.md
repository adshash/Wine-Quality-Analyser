# Wine-Quality-Analyser
Libraries: Sklearn, numpy, pandas, seaborn, xgboost and matplotlib.
This is a simple classification project that predicts the quality of wine based on several chemical factors. The program is made to be scalable but the data set being used was downloaded from kraggle. The quality will be predicted by several machine learning models and each model will be evaluated in order to decern which is best.

A heatmap is used to find correlation between variables to simplify the data in preprocessing. 3 Models are run: LogisticRegression, XGBClassifier and SVC. Their training accuracy and validation accuracy are outputed and the model with the smallest difference between these 2 numbers is used as predictor model. 

The best model has its confusion matrix outputted and displayed to give the user a form of visualisation as to how accurate of a model it is (not to brag). Then its classification report is outputted to give more info about its accuracy.

Then I used a wine example and the best model to predict whether or not the example would be of best quality or not.

I made this project to get a better understanding of machine learning classification, training and validation splits and visualisating the accuracy of different models.
