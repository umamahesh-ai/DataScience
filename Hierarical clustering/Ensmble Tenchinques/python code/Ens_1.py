
###Bagging/Boosting
import pandas as pd

df = pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/Ensamble Technique/Datasets_ET/Diabeted_Ensemble.csv")

# Dummy variables
df.head()
df.info()
list(df.columns)
# n-1 dummy variables will be created for n categories
df = pd.get_dummies(df, columns = [" Class variable"], drop_first = True)

df.head()


# Input and Output Split
predictors = df.loc[:, df.columns!=" Class variable_YES"]
type(predictors)

target = df[" Class variable_YES"]
type(target)

# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=0)

from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(learning_rate = 0.02, n_estimators = 5000)

ada_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(y_test, ada_clf.predict(x_test))
accuracy_score(y_test, ada_clf.predict(x_test))

# Evaluation on Training Data
accuracy_score(y_train, ada_clf.predict(x_train))



#########################################Stacking
# Libraries and data loading

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np

# Create the ensemble's base learners and meta-learner
# Append base learners to a list
base_learners = []

# KNN regression model
knn = KNeighborsRegressor(n_neighbors=5)
base_learners.append(knn)

# Decision Tree regressor model
dtr = DecisionTreeRegressor(max_depth=4, random_state=123456)
base_learners.append(dtr)

# Ridge regression
ridge = Ridge()
base_learners.append(ridge)

# Meta model using linear regerssion model
meta_learner = LinearRegression()


# Create the training metadata
# Create variables to store metadata and the targets

meta_data = np.zeros((len(base_learners), len(x_train)))
meta_targets = np.zeros(len(x_train))

# Create the cross-validation folds
KF = KFold(n_splits = 5)
meta_index = 0

for train_indices, test_indices in KF.split(x_train):
  # Train each learner on the K-1 folds 
  # and create metadata for the Kth fold
  for i in range(len(base_learners)):
    learner = base_learners[i]
    learner.fit(x_train[train_indices], y_train[train_indices])
    predictions = learner.predict(x_train[test_indices])
    meta_data[i][meta_index:meta_index + len(test_indices)] = predictions
    meta_targets[meta_index:meta_index + len(test_indices)] = y_train[test_indices]
    meta_index += len(test_indices)

# Transpose the metadata to be fed into the meta-learner
meta_data = meta_data.transpose()


# Create the metadata for the test set and evaluate the base learners
test_meta_data = np.zeros((len(base_learners), len(x_test)))
base_errors = []
base_r2 = []
for i in range(len(base_learners)):
  learner = base_learners[i]
  learner.fit(x_train, y_train)
  predictions = learner.predict(x_test)
  test_meta_data[i] = predictions

  err = metrics.mean_squared_error(y_test, predictions)
  r2 = metrics.r2_score(y_test, predictions)

  base_errors.append(err)
  base_r2.append(r2)

test_meta_data = test_meta_data.transpose()

# Fit the meta-learner on the train set and evaluate it on the test set
meta_learner.fit(meta_data, meta_targets)
ensemble_predictions = meta_learner.predict(test_meta_data)

err = metrics.mean_squared_error(y_test, ensemble_predictions)
r2 = metrics.r2_score(y_test, ensemble_predictions)

# Print the results 
for i in range(len(base_learners)):
  learner = base_learners[i]
  print(f'{base_errors[i]:.1f} {base_r2[i]:.2f} {learner.__class__.__name__}')
print(f'{err:.1f} {r2:.2f} Ensemble')



#############################################Voting

# Import the required libraries
from sklearn import datasets, linear_model, svm, neighbors, naive_bayes
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
breast_cancer = datasets.load_breast_cancer()
x, y = breast_cancer.data, breast_cancer.target

# Split the train and test samples
test_samples = 100
x_train, y_train = x[:-test_samples], y[:-test_samples]
x_test, y_test = x[-test_samples:], y[-test_samples:]

# Instantiate the learners (classifiers)
learner_1 = neighbors.KNeighborsClassifier(n_neighbors=5)
learner_2 = linear_model.Perceptron(tol=1e-2, random_state=0)
learner_3 = svm.SVC(gamma=0.001)

# Instantiate the voting classifier
voting = VotingClassifier([('KNN', learner_1),
                           ('Prc', learner_2),
                           ('SVM', learner_3)])

# Fit classifier with the training data
voting.fit(x_train, y_train)

# Predict the most voted class
hard_predictions = voting.predict(x_test)

# Accuracy of hard voting
print('Hard Voting:', accuracy_score(y_test, hard_predictions))

#################

# Soft Voting # 
# Instantiate the learners (classifiers)
learner_4 = neighbors.KNeighborsClassifier(n_neighbors = 5)
learner_5 = naive_bayes.GaussianNB()
learner_6 = svm.SVC(gamma = 0.001, probability = True)

# Instantiate the voting classifier
voting = VotingClassifier([('KNN', learner_4),
                           ('NB', learner_5),
                           ('SVM', learner_6)],
                            voting = 'soft')

# Fit classifier with the training data
voting.fit(x_train, y_train)
learner_4.fit(x_train, y_train)
learner_5.fit(x_train, y_train)
learner_6.fit(x_train, y_train)
# Predict the most probable class
soft_predictions = voting.predict(x_test)

# Get the base learner predictions
predictions_4 = learner_4.predict(x_test)
predictions_5 = learner_5.predict(x_test)
predictions_6 = learner_6.predict(x_test)

# Accuracies of base learners
print('L4:', accuracy_score(y_test, predictions_4))
print('L5:', accuracy_score(y_test, predictions_5))
print('L6:', accuracy_score(y_test, predictions_6))

# Accuracy of Soft voting
print('Soft Voting:', accuracy_score(y_test, soft_predictions))

