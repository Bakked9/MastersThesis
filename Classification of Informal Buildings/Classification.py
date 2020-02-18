# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 02:16:09 2020

@author: BRENDA
"""

import pydot
import pandas as pd
import numpy as np
import seaborn as sns
sns.set()

import matplotlib
import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

import sklearn
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz, DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.exceptions import NotFittedError

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier

from IPython.display import display

#Some useful functions we'll use in this notebook
def display_confusion_matrix(target, prediction, score=None):
    cm = metrics.confusion_matrix(target, prediction)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    if score:
        score_title = 'Accuracy Score: {0}'.format(round(score, 5))
        plt.title(score_title, size = 14)
    classification_report = pd.DataFrame.from_dict(metrics.classification_report(target, prediction, output_dict=True), orient='index')
    display(classification_report.round(2))
  

# Importing the dataset
dataset = pd.read_csv('BT.csv')
X = dataset.drop(['OBJECTID', 'Cluster'], axis=1)
y = dataset['Cluster']




# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#################Logistic Regression CLASSIFIER 

# Create and train model on train data sample
lg = LogisticRegression(solver='lbfgs', random_state=42)
lg.fit(X_train, y_train)

#Predict for test data sample
logistic_prediction = lg.predict(X_test)

#Compute error between predicted data and true response and display it in confusion matrix
score = metrics.accuracy_score(y_test, logistic_prediction)
display_confusion_matrix(y_test, logistic_prediction, score=score)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,logistic_prediction))
print(classification_report(y_test,logistic_prediction))

################Decision Tree

dt = DecisionTreeClassifier(min_samples_split=15, min_samples_leaf=20, random_state=42)
dt.fit(X_train, y_train)
dt_prediction = dt.predict(X_test)

score = metrics.accuracy_score(y_test, dt_prediction)
display_confusion_matrix(y_test, dt_prediction, score=score)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,dt_prediction))
print(classification_report(y_test,dt_prediction))

############### SVM CLASSIFIER

svm = SVC(gamma='auto', random_state=42)
svm.fit(X_train, y_train)
svm_prediction = svm.predict(X_test)

score = metrics.accuracy_score(y_test, svm_prediction)
display_confusion_matrix(y_test, svm_prediction, score=score)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,svm_prediction))
print(classification_report(y_test,svm_prediction))

############# RANDOM fOREST CLASSIFIER
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_prediction = rf.predict(X_test)


score = metrics.accuracy_score(y_test, rf_prediction)
display_confusion_matrix(y_test, rf_prediction, score=score)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,rf_prediction))
print(classification_report(y_test,rf_prediction))



def build_ann(optimizer='adam'):
    
    # Initializing our ANN
    model = Sequential()
    
    # Adding the input layer and the first hidden layer of our ANN with dropout
    model.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu',input_dim = 3))

    # Adding the second hidden layer
    model.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu'))

    # Adding the output layer
    model.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

    # Compiling the ANN
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model
    
    # Initializing our ANN
    ann = Sequential()
    
    # Adding the input layer and the first hidden layer of our ANN with dropout
    ann.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu', input_dim = 3))
    
    # Adding the second hidden layer
    ann.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu'))

    # Adding the output layer
    ann.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

    # Compiling the ANN
    ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    
    return ann


opt = optimizers.Adam(lr=0.001)
ann = build_ann(opt)

 #Fitting the ANN to the Training set
ann.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

 #Predicting the Test set results
ann_prediction = ann.predict(X_test)
ann_prediction = (ann_prediction > 0.5)

score = metrics.accuracy_score(y_test, ann_prediction)
display_confusion_matrix(y_test, ann_prediction, score=score)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,ann_prediction))
print(classification_report(y_test,ann_prediction))


######################K-FOLD CROSS VALIDATION#######################

n_folds = 10
cv_score_lg = cross_val_score(estimator=lg, X=X_train, y=y_train, cv=n_folds, n_jobs=-1)
cv_score_dt = cross_val_score(estimator=dt, X=X_train, y=y_train, cv=n_folds, n_jobs=-1)
cv_score_svm = cross_val_score(estimator=svm, X=X_train, y=y_train, cv=n_folds, n_jobs=-1)
cv_score_rf = cross_val_score(estimator=rf, X=X_train, y=y_train, cv=n_folds, n_jobs=-1)
cv_score_ann = cross_val_score(estimator=KerasClassifier(build_fn=build_ann, batch_size=16, epochs=20, verbose=0), X=X_train, y=y_train, cv=n_folds, n_jobs=-1)

cv_result = {'lg': cv_score_lg, 'dt': cv_score_dt, 'svm': cv_score_svm, 'rf': cv_score_rf, 'ann': cv_score_ann}
cv_data = {model: [score.mean(), score.std()] for model, score in cv_result.items()}
cv_df = pd.DataFrame(cv_data, index=['Mean_accuracy', 'Variance'])
cv_df

plt.figure(figsize=(20,8))
plt.plot(cv_result['lg'])
plt.plot(cv_result['dt'])
plt.plot(cv_result['svm'])
plt.plot(cv_result['rf'])
plt.plot(cv_result['ann'])
plt.title('Models Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Trained fold')
plt.xticks([k for k in range(n_folds)])
plt.legend(['logreg', 'tree', 'randomforest', 'ann', 'svm'], loc='upper left')
plt.show()

#####################TEST ON NEW DATASET

test_df_raw = pd.read_csv('Test.csv')
test = test_df_raw.copy()
test.head()

# Create and train model on train data sample
model_test = ann()
model_test.fit(X, y)

# Predict for test data sample
prediction = model_test.predict(test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,ann_prediction))
print(classification_report(y_test,ann_prediction))

result_df = test_df_raw.copy()
result_df['Cluster'] = prediction
result_df.to_csv('Test.csv', columns=['Cluster'], index=False)

