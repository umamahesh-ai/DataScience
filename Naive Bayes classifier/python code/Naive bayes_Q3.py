

#import libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

#loading dataset
tweets = pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/Naive Bayes/Datasets_Naive Bayes/Disaster_tweets_NB.csv",encoding = "utf-8")
tweets.isna().sum()
tweets.info()
tweets = tweets.iloc[:,3:]

# cleaning the data 
import re
def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+", " ",i).lower()
    i = re.sub("[0-9" "]+", " ", i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))
tweets.text=tweets.text.apply(cleaning_text)

# removing empty rows
tweets = tweets.loc[tweets.text != " ",:]

#split the data
from sklearn.model_selection import train_test_split

tw_train,tw_test=train_test_split(tweets,test_size=0.2)

# creating a matrix of token counts for the entire text document 
def split_into_words(i):
    return [word for word in i.split(" ")]

# Defining the preparation of email texts into word count matrix format - Bag of Words
tw_bow = CountVectorizer(analyzer = split_into_words).fit(tweets.text)

# Defining BOW for all messages
all_tw_matrix = tw_bow.transform(tweets.text)

# For training messages
train_tw_matrix = tw_bow.transform(tw_train.text)

# For testing messages
test_tw_matrix =tw_bow.transform(tw_test.text)

# Learning Term weighting and normalizing on entire emails
tfidf_transformer = TfidfTransformer().fit(all_tw_matrix)

# Preparing TFIDF for train emails
train_tfidf = tfidf_transformer.transform(train_tw_matrix)
train_tfidf.shape # (row, column)

# Preparing TFIDF for test emails
test_tfidf = tfidf_transformer.transform(test_tw_matrix)
test_tfidf.shape #  (row, column)

# Preparing a naive bayes model on training data set 
from sklearn.naive_bayes import MultinomialNB as MB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_tfidf, tw_train.target)

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(test_tfidf)

#find accuracy
from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m, tw_test.target) 
pd.crosstab(test_pred_m, tw_test.target)
