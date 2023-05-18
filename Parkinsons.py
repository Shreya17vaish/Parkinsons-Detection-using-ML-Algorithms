#importing all the important librarys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# StandardScaler function is used to standardize data in a comman range to easily understand it.
from sklearn.preprocessing import StandardScaler
from sklearn import svm # Support vector machine
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score # give accuracy

#loading data from csv to a pandas dataframe
parkinson_data = pd.read_csv(r"parkinsons.csv")

# printing first 5 rows of dataframe
parkinson_data.head()

#loading data last 5 rows of dataframe
parkinson_data.tail()

#number of rows and columns in dataframe
parkinson_data.shape

#getting more info about data
parkinson_data.info()

#getting some staistical measures about data
parkinson_data.describe()

#distribution of target column
parkinson_data['status'].value_counts()

#grouping data based on status
parkinson_data.groupby('status').mean()

#making pie chart and sharing the status report of parkinson and healthy person in pie graph from usin pyplot library
parkinson_data['status'].value_counts().plot(kind='pie',autopct="%1.0f%%")

# Split dataset into features (X) and target (y)
# target_column is status because we have to check using the voice recording sample of the patients
#whether they are having parkinson's or not
X = parkinson_data.copy()
X = X.drop(columns = ['name','status'], axis=1)
y = parkinson_data['status']
print(X)
print(y)


#making boxplot chart
parkinson_data.boxplot(figsize=(15,7))

#split dataset into training and testing sets
#here , training data is 80% and testing data is 20%
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)

#getting shape checked again
print(X.shape,X_train.shape,X_test.shape)
print(y.shape,y_train.shape,y_test.shape)

#Data standardization : to make data in same range but it wouldn't chnge the meaning of data
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
print(X_train)

#SVC(Support vecxtor Classifier)classifies the data
model=svm.SVC(kernel="linear",C=1,gamma='scale')
#training svm model with training data
model.fit(X_train,y_train)

#Evaluating SVM on Training data to find accuracy score
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(y_train,X_train_prediction)
print("Accuracy score of training data :",training_data_accuracy*100)

X_test_prediction=model.predict(X_test)
testing_data_accuracy=accuracy_score(y_test,X_test_prediction)
print("Accuracy score of testing data :",testing_data_accuracy*100)


#Initialize  a Random Forest Classifier
rfc=RandomForestClassifier(n_estimators=100,random_state=2)
#Fit the model on the training data
rfc.fit(X_train,y_train)
#Make predictions on the testing data
y_pred=rfc.predict(X_test)
#Calculate the accuracy of the model
testing_accuracy=accuracy_score(y_test,y_pred)
print("Accuracy of testing data :",testing_accuracy*100)

X_train_prediction =rfc.predict(X_train)
training_data_accuracy=accuracy_score(y_train,X_train_prediction)
print("Accuracy score of training data :",training_data_accuracy*100)

#create a KNN classifier with k=3
knn=KNeighborsClassifier(n_neighbors=3)
#Train the classifier on the training data
knn.fit(X_train,y_train)
#Evaluate the accuracy of the classifier
y_pred=knn.predict(X_test)

print("Accuracy of testing data of KNN algorithm :",(accuracy_score(y_test,y_pred))*100)

X_train_prediction=knn.predict(X_train)
training_data_accuracy=accuracy_score(y_train,X_train_prediction)
print("Accuracy score of training data :",training_data_accuracy*100)

#Histogram graph
parkinson_data.hist()

#Checking for missing values in each column
parkinson_data.isnull().sum()








