
import pandas as pd

train_data = pd.read_csv('C:/Users/HAI/Desktop/360DigitMG/Assingment/Naive Bayes/Datasets_Naive Bayes/SalaryData_Train.csv', encoding = 'utf-8')
test_data = pd.read_csv('C:/Users/HAI/Desktop/360DigitMG/Assingment/Naive Bayes/Datasets_Naive Bayes/SalaryData_Test.csv', encoding = 'utf-8')

train_data.isna().sum()
test_data.isna().sum()

train_data.columns
train_x = train_data.iloc[:,1:13].values
train_y = train_data["Salary"].values

test_x = test_data.iloc[:,1:13].values
test_y  = test_data["Salary"].values

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

#lable encoding for train data

train_x[:,0] = encoder.fit_transform(train_x[:,0])
train_x[:,1] = encoder.fit_transform(train_x[:,1])
train_x[:,3] = encoder.fit_transform(train_x[:,3])
train_x[:,4] = encoder.fit_transform(train_x[:,4])
train_x[:,5] = encoder.fit_transform(train_x[:,5])
train_x[:,6] = encoder.fit_transform(train_x[:,6])
train_x[:,7] = encoder.fit_transform(train_x[:,7])
train_x[:,11] = encoder.fit_transform(train_x[:,11])
train_y = encoder.fit_transform(train_y)

#lable encoding for test data

test_x[:,0] = encoder.fit_transform(test_x[:,0])
test_x[:,1] = encoder.fit_transform(test_x[:,1])
test_x[:,3] = encoder.fit_transform(test_x[:,3])
test_x[:,4] = encoder.fit_transform(test_x[:,4])
test_x[:,5] = encoder.fit_transform(test_x[:,5])
test_x[:,6] = encoder.fit_transform(test_x[:,6])
test_x[:,7] = encoder.fit_transform(test_x[:,7])
test_x[:,11] = encoder.fit_transform(test_x[:,11])
test_y = encoder.fit_transform(test_y)

#standardscaler

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

train_x = scaler.fit_transform(train_x)
test_x = scaler.fit_transform(test_x)

#train the naive bayes model to train

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(train_x,train_y)

#predict the y 
y = classifier.predict(test_x)

#finding accuracy
from sklearn.metrics import confusion_matrix, accuracy_score

confusion = confusion_matrix(test_y,y)
confusion

accuracy = accuracy_score(test_y,y)
accuracy
