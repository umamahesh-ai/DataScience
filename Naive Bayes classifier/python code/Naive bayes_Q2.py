
#import libraries
import pandas as pd
#loading dataset
car_ad = pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/Naive Bayes/Datasets_Naive Bayes/NB_Car_Ad.csv")
car_ad.isna().sum()
car_ad.info()

#label encoding
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
car_ad['Gender'] = encoder.fit_transform(car_ad['Gender'])

from sklearn.model_selection import train_test_split
train,test = train_test_split(car_ad, test_size = 0.20)
train_X = train.iloc[:, 1:4]
train_y = train.iloc[:, 4]
test_X  = test.iloc[:, 1:4]
test_y  = test.iloc[:, 4]

#train the naive bayas model to train
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(train_X,train_y)

#predict the y 
y=classifier.predict(test_X)
y

#finding accuracy
from sklearn.metrics import confusion_matrix,accuracy_score
confusion = confusion_matrix(test_y,y)
confusion

accuracy = accuracy_score(test_y,y)
accuracy
