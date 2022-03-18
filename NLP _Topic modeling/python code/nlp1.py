import pandas as pd
data=pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/NLP/NLP-TM/Data.csv")
data.describe()
data.info
data.isna()
data.isna().sum()
 
# drop the tweet created column , id does not give any information
data.drop(['tweet_created','tweet_id','sentiment','tweet_location','user_timezone'],axis=1,inplace=True)
data['text']=data['text'].apply(str)
# now we can do the text summerization of the text
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from string import punctuation
from heapq import nlargest

STOPWORDS = set(stopwords.words('english') + list(punctuation))
MIN_WORD_PROP, MAX_WORD_PROP = 0.1, 0.9

def compute_word_frequencies(data):
    words = [word for sentence in data 
                     for word in sentence 
                         if word not in STOPWORDS]
    counter = Counter(words)
    limit = float(max(counter.values()))
    word_frequencies = {word: freq/limit 
                                for word,freq in counter.items()}
    # Drop words if too common or too uncommon
    word_frequencies = {word: freq 
                            for word,freq in word_frequencies.items() 
                                if freq > MIN_WORD_PROP 
                                and freq < MAX_WORD_PROP}
    return word_frequencies
####

####
def summarize(text:str, num_sentences=3):
    """
    Summarize the text, by return the most relevant sentences
     :text the text to summarize
     :num_sentences the number of sentences to return
    """
    text = text.lower() # Make the text lowercase
    
    sentences = sent_tokenize(data) # Break text into sentences 
    
    # Break sentences into words
    word_sentences = [word_tokenize(sentence) for sentence in sentences]
    
    # Compute the word frequencies
    word_frequencies = compute_word_frequencies(data)
    
    # Calculate the scores for each of the sentences
    scores = [sentence_score(data, word_frequencies) for data in data]
    sentence_scores = list(zip(data, scores))
    
    # Rank the sentences
    top_sentence_scores = nlargest(data, sentence_scores, key=lambda t: t[1])
    
    # Return the top sentences
    return [t[0] for t in top_sentence_scores]
###
    

with open('C:/Users/HAI/Desktop/360DigitMG/Assingment/NLP/NLP-TM/Data.csv', 'r') as file:
    lor = file.read()

lor

len(sent_tokenize(lor))

summarize(lor)

summarize(lor, num_sentences=1)


