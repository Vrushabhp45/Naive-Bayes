import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Import dataset
salarydata_train = pd.read_csv('C:/Users/HP/PycharmProjects/Excelrdatascience/SalaryData_Train.csv')
salarydata_train.head()
salarydata_test = pd.read_csv('C:/Users/HP/PycharmProjects/Excelrdatascience/SalaryData_Test.csv')
salarydata_test.head()
#Exploratory data analysis
salarydata_train.shape
salarydata_test.shape
salarydata_train.info()
salarydata_train.describe()
salarydata_test.info()
salarydata_test.describe()
#Declare feature vector and target variable
X = salarydata_train.drop(['Salary'], axis=1)
y = salarydata_train['Salary']
#Split data into separate training and test set
# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
# display categorical variables
categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']
# display numerical variables
numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']
#Converting all the categorical data into numeric
cat_columns = X_train.select_dtypes(['object']).columns
#convert all categorical columns to numeric
X_train[cat_columns] = X_train[cat_columns].apply(lambda x: pd.factorize(x)[0])
X_train.head()
X_train.shape
cat_columns = X_test.select_dtypes(['object']).columns
#convert all categorical columns to numeric
X_test[cat_columns] = X_test[cat_columns].apply(lambda x: pd.factorize(x)[0])
X_test.head()
X_test.shape
#Feature Engineering
X_train.dtypes
X_test.dtypes
X_train.head()
X_test.head()
#Feature Scaling
cols = X_train.columns
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])
X_train.head()
#Model training
# train a Gaussian Naive Bayes classifier on the training set
from sklearn.naive_bayes import GaussianNB
# instantiate the model
gnb = GaussianNB()
# fit the model
gnb.fit(X_train, y_train)
#Predict the results
y_pred = gnb.predict(X_test)
y_pred
#Check accuracy score
from sklearn.metrics import accuracy_score
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
y_pred_train = gnb.predict(X_train)

y_pred_train
#Compare the train-set and test-set accuracy
y_pred_train = gnb.predict(X_train)

print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
#Confusion matrix
# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])
# visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
#Classification metrices
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
