# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 10:12:02 2021

@author: PUSPAK
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics

# Importing the Dataset
stroke = pd.read_csv('healthcare-dataset-stroke-data.csv')
stroke.head(10)
stroke.shape  #showing the shape of the data
stroke.columns  #showing the name of the columns
stroke.info()  #showing the data type for each of the columns
# Converting from float to object
stroke['stroke'] = stroke['stroke'].astype(object)
stroke['id'] = stroke['id'].astype(object)
stroke['hypertension'] = stroke['hypertension'].astype(object)
stroke['heart_disease'] = stroke['heart_disease'].astype(object)
stroke.info()
# Now we have 3 columns with numerical data, so we can see the description of those columns
stroke.describe()
#*****from the result of the above code we can see:
#*****average age = 43.23 with std = 22.612 and median = 45
#*****average glucose level = 106.147 with std = 45.28 and median = 91.885
#*****average bmi = 28.89 with std = 7.85 and median = 28.1

# Checking null values
stroke.isnull().sum()
#*****it is seen from the result of the above code that there is 201 null values in the column 'bmi' out of 5110
#*****entries
# Calculating Mode for the column 'bmi':
mode_bmi = statistics.mode(stroke['bmi'])
# Fill up 'na' values in 'bmi' column:
stroke['bmi'] = stroke['bmi'].fillna(mode_bmi)

# again checking for null values:
stroke.isnull().sum().sum() #the output came as 0, means all the null values have been filled up with mode

# We can see from the dataset the the column 'smoking_status' has one category called 'Unknown' which we'll treat as a 
# missing value. Out of 5110 number of entries we have 1543 records of 'Unknown' whhich is roughly 30% of the whole column.
# So, we need to impute them with the forward filling statement.
# First of all we need to replace all the 'Unknown' entries with np.NaN
for i in stroke['smoking_status']:
    if i == 'Unknown':
        stroke['smoking_status'].replace('Unknown', np.nan, inplace = True)

# Now forward fill all the missing values
stroke['smoking_status'] = stroke['smoking_status'].ffill(axis = 0)

# Dividing the features
# Categorical Features:
cat_feature = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status',
               'stroke']
# Numerical Features:
num_feature = ['age', 'bmi', 'avg_glucose_level']

# Let's find out some interesting distribution map of the numerical features with seaborn package:
for j in num_feature:
    sns.distplot(stroke[j])
    plt.show()

# As we can see we have someCategorical Features
# Let's plot them-
for i in cat_feature:
    sns.countplot(stroke[i])
    plt.show()

# Pair Plot visualization
sns.pairplot(stroke)
plt.show()
# Now we'd concentrate on the columns with categorical variables:
# For this we need 'sklearn' library to apply 

# We're going to use Label Encoding for all the categorical columns 

# Applying Label Encoding::
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


for k in stroke[cat_feature]:
    stroke[k] = le.fit_transform(stroke[k])


# Dividing the datasets into dependent & independent variables.
# Set of Independent variables will be maked as 'X' whereas that of dependent variable be 'y'

# As we can see the very first column of 'stroke' dataset i.e. 'i.d' has no effect on the prediction purpose. So, we can 
# omit this.
stroke_revised = stroke.loc[:, stroke.columns != 'id']

# creating 'X'
X = stroke_revised.iloc[:, :-1].values  # Independent Features

# Creating 'y'
y = stroke_revised.iloc[:, -1].values   # Dependent Features

# Diving the dataset into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)


# Applying Standardization on Numerical Features
#**** I use Standardization as the Distplot of the numerical features in the data seems to follow( roughly) the 
#**** Gaussian Distribution******#

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#************ Applying different Classification techniques ******************

# K-Nearest Neighbour Algorithm

# Training the K-NN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(X_train, y_train)

# Predicting the Test set results
y_pred_knn = knn.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)
print(cm_knn)

# Visualizing Confusion Matrix
ax= plt.subplot()
sns.heatmap(cm_knn, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('True', fontsize=20)
ax.xaxis.set_label_position('top') 
ax.xaxis.set_ticklabels(['Did not suffer stroke','suffered stroke'], fontsize = 10)
ax.xaxis.tick_top()

ax.set_ylabel('Predicted', fontsize=20)
ax.yaxis.set_ticklabels(['Did not suffer stroke', 'suffered stroke'], fontsize = 10)
plt.show()

# Accuracy Score for KNN
from sklearn.metrics import accuracy_score
accuracy_knn = accuracy_score(y_test, y_pred_knn)  # 94.84% is the accuracy score

# To know the accuracy doesn't come as a result of sheer good luck, we need to perform
# Applying k-Fold Cross Validation where K = 10
from sklearn.model_selection import cross_val_score
accuracies_knn = cross_val_score(estimator = knn, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies_knn.mean()*100))  # Accuracy = 94.74%
print("Standard Deviation: {:.2f} %".format(accuracies_knn.std()*100))  # Standard deviation = 0.27%

# Logistic Regression

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
logit = LogisticRegression(random_state = 0)
logit.fit(X_train, y_train)

# Predicting the Test set results
y_pred_logit = logit.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_logit = confusion_matrix(y_test, y_pred_logit)
print(cm_logit)

# Visualizing Confusion Matrix
ax= plt.subplot()
sns.heatmap(cm_logit, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('True', fontsize=20)
ax.xaxis.set_label_position('top') 
ax.xaxis.set_ticklabels(['Did not suffer stroke','suffered stroke'], fontsize = 10)
ax.xaxis.tick_top()

ax.set_ylabel('Predicted', fontsize=20)
ax.yaxis.set_ticklabels(['Did not suffer stroke', 'suffered stroke'], fontsize = 10)
plt.show()

# Accuracy Score for Logistic Regression
from sklearn.metrics import accuracy_score
accuracy_logit = accuracy_score(y_test, y_pred_logit)  # 94.97% is the accuracy score

# To know the accuracy doesn't come as a result of sheer good luck, we need to perform
# Applying k-Fold Cross Validation where K = 10
from sklearn.model_selection import cross_val_score
accuracies_logit = cross_val_score(estimator = logit, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies_logit.mean()*100))  # Accuracy = 95.14%
print("Standard Deviation: {:.2f} %".format(accuracies_logit.std()*100))  # Standard deviation = 0.14%

# Naive Bayes

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)

# Predicting the Test set results
y_pred_nb = nb.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_nb = confusion_matrix(y_test, y_pred_nb)
print(cm_nb)

# Visualizing Confusion Matrix
ax= plt.subplot()
sns.heatmap(cm_nb, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('True', fontsize=20)
ax.xaxis.set_label_position('top') 
ax.xaxis.set_ticklabels(['Did not suffer stroke','suffered stroke'], fontsize = 10)
ax.xaxis.tick_top()

ax.set_ylabel('Predicted', fontsize=20)
ax.yaxis.set_ticklabels(['Did not suffer stroke', 'suffered stroke'], fontsize = 10)
plt.show()

# Accuracy Score for Naive Bayes
from sklearn.metrics import accuracy_score
accuracy_nb = accuracy_score(y_test, y_pred_nb)  # 87.67% is the accuracy score

# To know the accuracy doesn't come as a result of sheer good luck, we need to perform
# Applying k-Fold Cross Validation where K = 10
from sklearn.model_selection import cross_val_score
accuracies_nb = cross_val_score(estimator = nb, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies_nb.mean()*100))  # Accuracy = 87%
print("Standard Deviation: {:.2f} %".format(accuracies_nb.std()*100))  # Standard deviation = 1.16%

# Decision Tree

# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
dc_tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dc_tree.fit(X_train, y_train)

# Predicting the Test set results
y_pred_dc_tree = dc_tree.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_dc_tree = confusion_matrix(y_test, y_pred_dc_tree)
print(cm_dc_tree)

# Visualizing Confusion Matrix
ax= plt.subplot()
sns.heatmap(cm_dc_tree, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('True', fontsize=20)
ax.xaxis.set_label_position('top') 
ax.xaxis.set_ticklabels(['Did not suffer stroke','suffered stroke'], fontsize = 10)
ax.xaxis.tick_top()

ax.set_ylabel('Predicted', fontsize=20)
ax.yaxis.set_ticklabels(['Did not suffer stroke', 'suffered stroke'], fontsize = 10)
plt.show()

# Accuracy Score for Decision Tree
from sklearn.metrics import accuracy_score
accuracy_dc_tree = accuracy_score(y_test, y_pred_dc_tree)  # 92.04% is the accuracy score

# To know the accuracy doesn't come as a result of sheer good luck, we need to perform
# Applying k-Fold Cross Validation where K = 10
from sklearn.model_selection import cross_val_score
accuracies_dc_tree = cross_val_score(estimator = dc_tree, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies_dc_tree.mean()*100))  # Accuracy = 91.19%
print("Standard Deviation: {:.2f} %".format(accuracies_dc_tree.std()*100))  # Standard deviation = 1.87%

# Random Forest

# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
rt = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rt.fit(X_train, y_train)

# Predicting the Test set results
y_pred_rt = rt.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_rt = confusion_matrix(y_test, y_pred_rt)
print(cm_rt)

# Visualizing Confusion Matrix
ax= plt.subplot()
sns.heatmap(cm_rt, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('True', fontsize=20)
ax.xaxis.set_label_position('top') 
ax.xaxis.set_ticklabels(['Did not suffer stroke','suffered stroke'], fontsize = 10)
ax.xaxis.tick_top()

ax.set_ylabel('Predicted', fontsize=20)
ax.yaxis.set_ticklabels(['Did not suffer stroke', 'suffered stroke'], fontsize = 10)
plt.show()

# Accuracy Score for Random Forest
from sklearn.metrics import accuracy_score
accuracy_rt = accuracy_score(y_test, y_pred_rt)  # 94.97% is the accuracy score

# To know the accuracy doesn't come as a result of sheer good luck, we need to perform
# Applying k-Fold Cross Validation where K = 10
from sklearn.model_selection import cross_val_score
accuracies_rt = cross_val_score(estimator = rt, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies_rt.mean()*100))  # Accuracy = 94.86%
print("Standard Deviation: {:.2f} %".format(accuracies_rt.std()*100))  # Standard Deviation = 0.5%

# Kernel SVM

# Training the Kernel SVM model on the Training set
from sklearn.svm import SVC
k_svm = SVC(kernel = 'rbf', random_state = 0)
k_svm.fit(X_train, y_train)

# Predicting the Test set results
y_pred_svm = k_svm.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_k_svm = confusion_matrix(y_test, y_pred_svm)
print(cm_k_svm)

# Visualizing Confusion Matrix
ax= plt.subplot()
sns.heatmap(cm_k_svm, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('True', fontsize=20)
ax.xaxis.set_label_position('top') 
ax.xaxis.set_ticklabels(['Did not suffer stroke','suffered stroke'], fontsize = 10)
ax.xaxis.tick_top()

ax.set_ylabel('Predicted', fontsize=20)
ax.yaxis.set_ticklabels(['Did not suffer stroke', 'suffered stroke'], fontsize = 10)
plt.show()

# Accuracy Score for Kernel SVM
from sklearn.metrics import accuracy_score
accuracy_k_svm = accuracy_score(y_test, y_pred_svm)  # 95.04% is the accuracy score


# To know the accuracy doesn't come as a result of sheer good luck, we need to perform
# Applying k-Fold Cross Validation where K = 10
from sklearn.model_selection import cross_val_score
accuracies_svc = cross_val_score(estimator = k_svm, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies_svc.mean()*100))  # Accuracy = 95.16 %
print("Standard Deviation: {:.2f} %".format(accuracies_svc.std()*100)) # standard Deviation = 0.13%


# After taking into consideration of all the models and their subsequent accuracies, we choose Kernel SVM
# as it has the highest Accuracy after K-Fold Cross Validation as well as lowest Standard Deviation

# Creating the output file (CSV)

id = [stroke['id']]
#original = [stroke['stroke']]
predicted = [y_pred_svm]

premature = [id,  predicted]

output_file = pd.DataFrame(data = premature)

output_file_revised = output_file.transpose(copy = True)