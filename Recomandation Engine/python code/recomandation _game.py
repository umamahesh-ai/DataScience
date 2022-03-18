
import pandas as pd

game = pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/Recomendation Engine/Datasets_Recommendation Engine/game.csv", encoding='utf-8')
game.head()
game.columns
game.game
 #term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus
from sklearn.feature_extraction.text import TfidfVectorizer 

# Creating a Tfidf Vectorizer to remove all stop words

tfidf = TfidfVectorizer(stop_words = 'english')

# replacing the NaN values in overview column with empty string

game['game'].isnull().sum()

# Preparing the Tfidf matrix by fitting and transforming

tfidf_matrix = tfidf.fit_transform(game.game)  #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape

# with the above matrix we need to find the similarity score
# There are several metrics for this such as the euclidean, 
# the Pearson and the cosine similarity scores

# For now we will be using cosine similarity matrix
# A numeric quantity to represent the similarity between 2 movies 
# Cosine similarity - metric is independent of magnitude and easy to calculate 

# cosine(x,y)= (x.y⊺)/(||x||.||y||)

from sklearn.metrics.pairwise import linear_kernel

# Computing the cosine similarity on Tfidf matrix

cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

game = game.drop(['userId'], axis = 1, inplace = True)

# creating a mapping of game name to index number 

game_index = pd.Series(game.index, index = game['game']).drop_duplicates()

game_index.duplicated().sum()

def get_recommendations(Name, topN):
    # topN = 10
    # getting the game index using its name
    game_id = game_index[Name]
    
    # getting the pairwise similarity scores for all the games
    cosine_scores = list(enumerate(cosine_sim_matrix[game_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key = lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar games 
    cosine_scores_N = cosine_scores[0: topN+1]
    
    # Getting the game index
    game_idx = [i[0] for i in cosine_scores_N]
    game_scores = [i[1] for i in cosine_scores_N]
    
    # Similar games and scores
    game_similar_show = pd.DataFrame(columns = ["game", "Score"])
    game_similar_show["game"] = game.loc[game_idx, "game"]
    game_similar_show["Score"] = game_scores
    game_similar_show.reset_index(inplace = True)
    # game_similar_show.drop(["index"], axis=1, inplace=True)
    print(game_similar_show)
    # return (game_similar_show)

# Enter your game and number of games to be recommended 

get_recommendations("The Legend of Zelda: Ocarina of Time", topN = 10)
game_index["The Legend of Zelda: Ocarina of Time"]












