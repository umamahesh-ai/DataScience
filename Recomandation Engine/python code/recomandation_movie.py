
import pandas as pd
movie = pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/Recomendation Engine/Datasets_Recommendation Engine/Entertainment.csv", encoding = 'utf-8')
movie.head()
movie.columns
movie.Category
#term frequencey-inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus
from sklearn.feature_extraction.text import TfidfVectorizer
# To remove all stop words in thre Tfidf vectorizer
tfidf = TfidfVectorizer(stop_words = 'english')
# check the null values in the category column
movie['Category'].isnull().sum()
# Apply the Tfidf on the column of category 
tfidf_matrix = tfidf.fit_transform(movie.Category)
tfidf_matrix.shape

from sklearn.metrics.pairwise import linear_kernel

cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

movie_index = pd.Series(movie.index, index = movie['Titles']).drop_duplicates()

def get_recommendations(Name, topN):
    movie_id =  movie_index [Name]
    cosine_scores = list(enumerate(cosine_sim_matrix[movie_id]))
    cosine_scores = sorted(cosine_scores, key = lambda x:x[1], reverse = True)
    cosine_scores_N = cosine_scores[0: topN+1]
    movie_idx = [i[0] for i in cosine_scores_N]
    movie_scores = [i[1] for i in cosine_scores_N]
    movie_similar_show = pd.DataFrame(columns = ['Titles', 'Scores'])
    movie_similar_show['Titles'] = movie.loc[movie_idx, 'Titles']
    movie_similar_show['Scores'] = movie_scores
    movie_similar_show.reset_index(inplace = True)
    print(movie_similar_show)

get_recommendations('Mortal Kombat (1995)', topN = 10)
movie_index['Mortal Kombat (1995)']
