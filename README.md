# Iris-Dataset
This python script runs the following classifiers on the popular Iris dataset and is for learning purposes.
- Logistic Regression (LR)
- Support Vector Machine (SVM)
- Linear Disciminant Analysis (LDA)
- Decision Tree (CART)
- K Nearest Neigbours (KNN)
- Gaussian Naive Bayes (GNB)

The dataset is loaded from the following URL:
```
https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
```
Plotting a scatter-matrix plot gives us a good idea of which columns correlate with each other

<img src="https://github.com/Swapnil52/Iris-Dataset/blob/master/Iris/iris.png?raw=true" height=300>

First, the dataset is split into training and test sets in a 80:20 ratio. Secondly, for cross-validation, 10-fold cross validation is performed and the classifier with the best accuracy is chosen for making predictions, which, in this case, is K-Nearest Neighbours (~98.33%). 

<img src="https://github.com/Swapnil52/Iris-Dataset/blob/master/Iris/iris_output.png?raw=true" height=300>
