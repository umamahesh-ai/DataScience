
import pandas as pd
import numpy as np
forestfires = pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/SVM/Datasets_SVM/forestfires.csv")
forestfires.describe()
forestfires.head()  
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
forestfires.drop(["month","day"],axis=1,inplace=True)
train_data,test_data = train_test_split(forestfires, test_size = 0.20)

#splitting the data
train_x=train_data.iloc[:,1:28].values
train_y=train_data['size_category'].values
test_x=test_data.iloc[:,1:28].values
test_y = test_data['size_category'].values

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

    