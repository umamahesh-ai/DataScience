

import pandas as pd
import numpy as np
# dataset have assing the train and test in saperate 
test_data = pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/SVM/Datasets_SVM/SalaryData_Test (1).csv")
train_data=pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/SVM/Datasets_SVM/SalaryData_Train (1).csv")
test_data.describe()
train_data.describe()
test_data.head()
train_data.head()
#finding na
train_data.isna().sum()
test_data.isna().sum()
#splitting the data
train_x=train_data.iloc[:,1:13].values
train_y=train_data['Salary'].values
test_x=test_data.iloc[:,1:13].values
test_y = test_data['Salary'].values
#lablem encoding for train data
from sklearn.preprocessing import LabelEncoder
la=LabelEncoder()
train_x[:,0] = la.fit_transform(train_x[:,0])
train_x[:,1] = la.fit_transform(train_x[:,1])
train_x[:,3] = la.fit_transform(train_x[:,3])
train_x[:,4] = la.fit_transform(train_x[:,4])
train_x[:,5] = la.fit_transform(train_x[:,5])
train_x[:,6] = la.fit_transform(train_x[:,6])
train_x[:,7] = la.fit_transform(train_x[:,7])
train_x[:,11] = la.fit_transform(train_x[:,11])
train_y=la.fit_transform(train_y)
#lable encoding for test data
test_x[:,0] = la.fit_transform(test_x[:,0])
test_x[:,1] = la.fit_transform(test_x[:,1])
test_x[:,3] = la.fit_transform(test_x[:,3])
test_x[:,4] = la.fit_transform(test_x[:,4])
test_x[:,5] = la.fit_transform(test_x[:,5])
test_x[:,6] = la.fit_transform(test_x[:,6])
test_x[:,7] = la.fit_transform(test_x[:,7])
test_x[:,11] = la.fit_transform(test_x[:,11])
test_y=la.fit_transform(test_y)
#standardscaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_x = sc.fit_transform(train_x)
test_x = sc.transform(test_x)

from sklearn.svm import SVC




# kernel = linear
help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(train_x, train_y)
pred_test_linear = model_linear.predict(test_x)

np.mean(pred_test_linear == test_y)

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_x, train_y)
pred_test_rbf = model_rbf.predict(test_x)

np.mean(pred_test_rbf==test_y)
###############confusion matrix and accuracy score
from sklearn.metrics import accuracy_score, confusion_matrix
# Evaluation on Testing Data,model_rbf
confusion_matrix(test_y, model_rbf.predict(test_x))
accuracy_score(test_y, model_rbf.predict(test_x))

# Evaluation on Training Data,model_rbf
accuracy_score(train_y, model_rbf.predict(train_x))


# Evaluation on Testing Data,model_linear
confusion_matrix(test_y, model_linear.predict(test_x))
accuracy_score(test_y, model_linear.predict(test_x))

# Evaluation on Training Data,model_linear
accuracy_score(train_y, model_linear.predict(train_x))
















