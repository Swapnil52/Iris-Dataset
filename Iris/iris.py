#python iris dataset analysis

import sys
import pandas
import matplotlib
import scipy
import numpy
import sklearn
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt 
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#load the iris csv file from UCI
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names = names)

scatter_matrix(dataset)
plt.show()

#now it is time to split the dataset into training and test sets for evaluation.
array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 7)

#Score different classifiers
seed = 7
scoring = 'accuracy'
lr = ('LR', LogisticRegression())
svm = ('SVM', SVC())
lda = ('LDA', LinearDiscriminantAnalysis())
knn = ('KNN', KNeighborsClassifier())
nb = ('NB', GaussianNB())
dt = ('CART', DecisionTreeClassifier())
models = [lr, svm, lda, knn, nb, dt]
names = []
results = []
for name, model in models:
	kfold = model_selection.KFold(n_splits = 10, random_state = 7)
	cv_result = model_selection.cross_val_score(model, X_train, Y_train, cv = kfold, scoring = scoring)
	results.append(cv_result)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_result.mean(), cv_result.std())
	print(msg)

#We see that KNeighbours has the best accuracy-98.33% so we will use it for predictions
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))





