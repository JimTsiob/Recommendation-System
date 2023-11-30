import pandas as pd # pip install pandas
import numpy as np # pip install numpy
import math
import sys
import nltk # pip install nltk
# nltk.download('stopwords') # <- run this if you don't have the stopwords already on your machine.
from nltk.corpus import stopwords
import re
from collections import Counter




# Metrics creation
def jaccard(a,b):
    """Define Jaccard Similarity function for two sets"""
    intersection = len(list(set(a).intersection(b)))
    union = (len(a) + len(b)) - intersection
    return float(intersection) / union

def dice(x,y):
    return (2*len(list(set(x).intersection(y)))) / (len(x) + len(y)) 

def cosine(a,b):
    n = len(a) # they're always same size since I get ratings of movies both users rated.
    numerator = 0
    denominator = 0
    epsilon = 1e-9 # this is to avoid divide by zero error, in case of same size same value vectors.
    sum1 = 0
    sum2 = 0
    root1 = 0
    root2 = 0
    for i in range(0,n):
        numerator += (a[i] * b[i])
        sum1 += a[i] ** 2
        sum2 += b[i] ** 2
        
    root1 = math.sqrt(sum1)
    root2 = math.sqrt(sum2)
    denominator = (root1 * root2) + epsilon
    return numerator / denominator


def pearson(x,y): 
    n = len(x) # they're always same size since I get ratings of movies both users rated.
    epsilon = 1e-9 # this is to avoid divide by zero error, in case of same size same value vectors.
    x_sum = sum(x)
    x_mean = x_sum / n
    y_sum = sum(y)
    y_mean = y_sum / n
    numerator = 0
    denominator = 0
    sum1 = 0
    sum2 = 0
    root1 = 0
    root2 = 0
    for i in range(0,n):
        numerator += (x[i] - x_mean) * (y[i] - y_mean)
        sum1 += (x[i] - x_mean) ** 2
        sum2 += (y[i] - y_mean) ** 2

    root1 = math.sqrt(sum1)
    root2 = math.sqrt(sum2)
    denominator = (root1 * root2) + epsilon
    return numerator / denominator

a = {1, 2, 3, 4}
b = {3, 4, 5, 6}
x = [1,2,3,4]
y = [3,4,5,6]
z = [1,2,3,4]
ena = [4.0]
dyo = [3.5,6.0,4.0]


def normalizer(x):
    # x: Iterable (most likely list)
    # normalizer to transform the ratings in the interval [0,1]
    epsilon = 1e-9 # this is to avoid divide by zero error, in case of same size same value vectors.
    min_x = min(x)
    max_x = max(x)
    norm_x = [(x_item - min_x) / (max_x - min_x + epsilon)  for x_item in x]
    return norm_x

def normalizeNum(x,min,max): # used for each separate rating in recommendation score calculation
    num = 0
    num = (x-min) / (max - min)
    return num


def userToUser(id,simFunc,k,directory):
    # id: id of the user for recommendation
    # simFunc: the similarity function used
    # k: k users with ratings most similar to user for recommendation

    # load datasets
    ratings_df = pd.read_csv(directory + '/ratings.csv')
    
    # genome_tags_df = pd.read_csv(directory + '/genome-tags.csv') # you can only load these two with the full dataset
    # genome_scores_df = pd.read_csv(directory + '/genome-scores.csv')

    user_x_ratings = ratings_df[ratings_df['userId'] == id]  # get ratings of x user
    user_x_movieIds = user_x_ratings['movieId'].unique() # get movie ids of user x to filter them out later on
    movie_ratings_y_users = ratings_df[(ratings_df['movieId'].isin(user_x_ratings['movieId'])) & (ratings_df['userId'] != id)] # get all ratings of users y for the same movies as x
    

    rx = {} # dictionary to store recommendation scores key: movieId , value: score
    
    x_rating_dict = dict(zip(user_x_ratings['movieId'], user_x_ratings['rating']))
    x_movies = list(x_rating_dict.keys()) # we'll use the movies of user x compared with each user's movies with similar movies for Jaccard and Dice.
    # find similarity scores for all users y with user x
    similarity_scores = {} # dictionary to hold sim scores , key: userId value: similarity score
    for userId in movie_ratings_y_users['userId']:
        y_similar_movie_rating = movie_ratings_y_users[movie_ratings_y_users['userId'] == userId] # get ratings for each user y for the same movies that x has watched

        x = []
        y = [] # these two lists will hold the ratings of user x and user y.
        # the dictionary below is used to keep only the ratings of the movies that both x and y have watched.
        y_rating_dict = dict(zip(y_similar_movie_rating['movieId'], y_similar_movie_rating['rating'])) # y rating dictionary, key: movieId , value: rating
        common_ratings = set(x_rating_dict.keys()).intersection(y_rating_dict.keys())
        y_movies = list(y_rating_dict.keys()) 
        common_ratings_dict = {key: (x_rating_dict[key], y_rating_dict[key]) for key in common_ratings}
        for val in common_ratings_dict.values():
            x.append(val[0])
            y.append(val[1])

        sxy = 0 # similarity score
        if simFunc.lower() == "jaccard":
            sxy = jaccard(x_movies,y_movies)
        elif simFunc.lower() == "dice":
            sxy = dice(x_movies,y_movies)
        elif simFunc.lower() == "cosine":
            x = normalizer(x)
            y = normalizer(y)
            sxy = cosine(x,y)
        else:
            # pearson handles normalization on it's own
            sxy = pearson(x,y)
        
        similarity_scores[userId] = sxy
    
    sorted_sxy = dict(sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)) # sort similarity scores in descending order
    k_users_similarity_score_dict = {key: sorted_sxy[key] for key in list(sorted_sxy)[:k]} # get this for recommendation score calc, will help in getting each different sim score for the numerator and all sim scores for denominator.
    sxy_sum = sum(k_users_similarity_score_dict.values()) # sum which will be used as denominator in recommendation score calculation
    first_k_users = list(sorted_sxy.keys())[:k] # get k most similar users
    y_user_ratings_different_movies_from_x = ratings_df[(ratings_df['userId'].isin(first_k_users)) & (~ratings_df['movieId'].isin(user_x_movieIds))] # get ratings of k most similar users y to x for other movies that x has not watched.
    
    # calculation of formula for recommendation score rxi for each movie not watched by user x
    denominator = sxy_sum
    numerator_sums = dict(zip(y_user_ratings_different_movies_from_x['movieId'],[0.0] * len(y_user_ratings_different_movies_from_x['movieId']))) # dictionary for numerator sums key: movieId, value:sum of sxy * ryi
    for index, row in y_user_ratings_different_movies_from_x.iterrows():
        y_user = row['userId']
        y_movie = row['movieId']
        y_rating = row['rating'] # get ratings for each movie
        y_rating_norm = normalizeNum(y_rating,0.0,5.0)
        # y = []
        # for rating in y_rating['rating']:
        #     y.append(rating)

        numerator = k_users_similarity_score_dict[y_user] * y_rating_norm # sxy * ryi
        numerator_sums[y_movie] += numerator


    for key in numerator_sums.keys():
        rxi = numerator_sums[key] / denominator
        rx[key] = rxi
    
    
    return rx
        
    

def itemToItem(id,simFunc,k,directory):
    # logic here is to find all similarity scores for pairs of movies that user x has watched
    # and movies that user x hasn't watched. Get the k most similar pairs (highest similarity score) 
    # then predict the ratings of user x for the other movies
    # (calculate recommendation score essentially) and print out the top n recommendations.

    # load datasets
    ratings_df = pd.read_csv(directory + '/ratings.csv')

    user_x_ratings = ratings_df[ratings_df['userId'] == id] # get ratings of user x
    user_x_movieIds = user_x_ratings['movieId'].unique() # get movie ids of user x to filter them out later on
    movie_ratings_y_users = ratings_df[(ratings_df['movieId'].isin(user_x_ratings['movieId'])) & (ratings_df['userId'] != id)]
    movie_ratings_for_x_movies = ratings_df[ratings_df['movieId'].isin(user_x_ratings['movieId'])] # get all x movie ratings for iteration below
    user_y_ids = movie_ratings_y_users['userId'].unique() # get y user ids to get the movies that x hasn't watched
    y_user_movies_other_than_x = ratings_df[(ratings_df['userId'].isin(user_y_ids)) & (~ratings_df['movieId'].isin(user_x_movieIds))]
    
    # # make pivot table from movies of similar users to x which
    # pivot_table = y_user_movies_other_than_x.pivot_table(index='userId', columns='movieId', values='rating',fill_value=0)
    pivot_table = y_user_movies_other_than_x.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0).T
    ratings = {movie: pivot_table.loc[movie].tolist() for movie in pivot_table.index}
    # pivot_table = pivot_table.T # transpose so movies are the indexes
 
    # ratings = {movie: pivot_table.loc[movie].tolist() for movie in pivot_table.index} # get all ratings from pivot
    # # mean-centering
    # # mean_centered_pivot = pivot_table.sub(pivot_table.mean(axis=1), axis=0)


    # similarity_scores = {}
    # for x_movieId in user_x_ratings['movieId']:
    #     x_movie_ratings = ratings_df[ratings_df['movieId'] == x_movieId]
    #     x = []
    #     for rating in x_movie_ratings['rating']:    
    #         x.append(rating)

    #     x_movie_users = []        
    #     for userId in x_movie_ratings['userId']:
    #         x_movie_users.append(userId)
    #     x_movie_users.append(id) # get all users who have watched movie x for jaccard and dice

    #     for i, y_movieId in enumerate(pivot_table.index):
    #         y = ratings[y_movieId]
    #         # optimised_y = [rating > 0 for rating in y]
    #         y_movie_df = ratings_df[ratings_df['movieId'] == y_movieId]
    #         y_movie_users = []
    #         for userId in y_movie_df['userId']:
    #             y_movie_users.append(userId)
             
    #         sxy = 0 # similarity score
    #         if simFunc.lower() == "jaccard":
    #             sxy = jaccard(x_movie_users,y_movie_users)
    #         elif simFunc.lower() == "dice":
    #             sxy = dice(x_movie_users,y_movie_users)
    #         elif simFunc.lower() == "cosine":
    #             x = normalizer(x)
    #             y = normalizer(y)
    #             sxy = cosine(x,y)
    #         else:
    #             # pearson handles normalization on it's own
    #             sxy = pearson(x,y)
            
    #         similarity_scores[(x_movieId,y_movieId)] = sxy
    #     print('done for movie: ', x_movieId)

    similarity_scores = {}

    # For every movie of x , take the similarity score with every other movie watched by y users who watched the same movies as x 
    for x_movieId, x_movie_ratings in movie_ratings_for_x_movies.groupby('movieId'):
        x = x_movie_ratings['rating'].tolist()

        x_movie_users = set(x_movie_ratings['userId'].tolist() + [id]) # get all users who have watched movie x for jaccard and dice
        
        # print('x_movieId: ',x_movieId)

        for y_movieId, y_movie_ratings in ratings_df[ratings_df['movieId'].isin(pivot_table.index)].groupby('movieId'):
            y = ratings[y_movieId]
            
            sxy = 0
            if simFunc.lower() == "jaccard":
                sxy = jaccard(x_movie_users, set(y_movie_ratings['userId'].tolist()))
            elif simFunc.lower() == "dice":
                sxy = dice(x_movie_users, set(y_movie_ratings['userId'].tolist()))
            elif simFunc.lower() == "cosine":
                x = normalizer(x)
                y = normalizer(y)
                sxy = cosine(x, y)
            else:
                sxy = pearson(x, y)

            similarity_scores[(x_movieId, y_movieId)] = sxy

    sorted_sxy = dict(sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)) # sort similarity scores in descending order
    k_movies_similarity_score_dict = {key: sorted_sxy[key] for key in list(sorted_sxy)[:k]} # get k most similar movies
    most_similar_movie_Ids = [] # get k most similar movie Ids to the movies user x has watched so we can check each of them later on
    for key in k_movies_similarity_score_dict.keys():
        most_similar_movie_Ids.append(key[1]) # take opposite key since this is the similar movie 

    most_similar_movie_df = ratings_df[ratings_df['movieId'].isin(most_similar_movie_Ids)] # filter out all other movies apart from the k most similar

    recommendation_scores = {} # recommendation score dictionary , Key: movieId Value: recommendation score
    for movieId in most_similar_movie_df['movieId']:
        # find recommendation scores for all movies that x hasn't watched.
        numerator = 0
        denominator = 0
        rxi = 0
        for index,row in user_x_ratings.iterrows():
            x_rating = row['rating']
            x_movieId = row['movieId']
            for key,value in k_movies_similarity_score_dict.items():
                if (x_movieId in key) & (movieId in key):
                    numerator += x_rating * k_movies_similarity_score_dict[(x_movieId,movieId)] # rating of movie x has watched * similarity_score(movie x has watched,movie to be predicted)
                    denominator += k_movies_similarity_score_dict[(x_movieId,movieId)] # sum of similarity scores of movies that x has watched with the movie to be predicted
                else:
                    continue
        rxi = numerator / denominator
        recommendation_scores[movieId] = rxi

    return recommendation_scores



def tagBasedRecommendation(id,simFunc,directory):
    # id: id of movie
    # simFunc: similarity metric
    # n: number of recommendations
    # directory: the directory to load the datasets from

    # Here, we get the tags for each movie, get their counts, compare them and get the most similar movies based on tag count.
    # for Jaccard and Dice we get the tags that have tag count >= 1

    # load datasets
    ratings_df = pd.read_csv(directory + '/ratings.csv')
    tags_df = pd.read_csv(directory + '/tags.csv')
    movies_df = pd.read_csv(directory + '/movies.csv')
    links_df = pd.read_csv(directory + '/links.csv')
    # genome_tags_df = pd.read_csv(directory + '/genome-tags.csv') # you can only load these two with the full dataset
    # genome_scores_df = pd.read_csv(directory + '/genome-scores.csv')

    print('ratings_df size: ', ratings_df.shape)
    print('tags_df size: ', tags_df.shape)
    print('movies_df size: ', movies_df.shape)
    print('links_df size: ', links_df.shape)
    # print('genome_tags_df size: ', genome_tags_df.shape)
    # print('genome_scores_df size: ', genome_scores_df.shape)
    # movies to check: 60756 -> comedy, 4343, 2, 31658

    
    tags = []
    for tag in tags_df['tag']:
        tag = tag.lower()
        tags.append(tag)

    tag_count_keys = []
    for index,row in tags_df.iterrows():
        movie = row['movieId']
        tag = row['tag']
        tag = tag.lower()
        tag_count_keys.append((movie,tag)) # get movie tag tuple to add as key on dictionary later on

    tags_count_dict = {key: 0 for key in tag_count_keys} # dictionary that will hold tags of all movies. key: movieId value: tag
    

    for index,row in tags_df.iterrows():
        movie = row['movieId']
        tag = row['tag']
        tag = tag.lower()
        tags_count_dict[movie,tag] += 1
    
    x_movie_tag_dict = {tag: 0 for tag in tags} # dictionary for tags of wanted movie (I'm calling it x movie). key: tag , value: count of tag
    other_than_x_tag_df = tags_df[tags_df['movieId'] != id] # dataframe with all other movie tags other than the wanted one.

    for key in tags_count_dict.keys():
        if key[0] == id:
            x_movie_tag_dict[key[1]] = tags_count_dict[(key[0],key[1])]

    similarity_scores = {} # dictionary to hold sim scores , key: userId value: similarity score
    for movieId in other_than_x_tag_df['movieId']:
        other_than_x_tag_dict = {tag: 0 for tag in tags} # dictionary for tags of all other movies. key: tag, value: count of tag
        for key in tags_count_dict.keys():
            if key[0] == movieId:
                other_than_x_tag_dict[key[1]] = tags_count_dict[(key[0],key[1])]

        # calculate similarity with tags of movie x with other movies' tags

        x = []
        y = []
        # creation of specific vectors for jaccard and dice, taking tags with counts >=1 instead of the counts
        x_jacc_dice = []
        y_jacc_dice = []
        for key in x_movie_tag_dict.keys():
            x.append(x_movie_tag_dict[key])
            if (x_movie_tag_dict[key] >= 1):
                x_jacc_dice.append(key)
            
        for key in other_than_x_tag_dict.keys():
            y.append(other_than_x_tag_dict[key])
            if (other_than_x_tag_dict[key] >= 1):
                y_jacc_dice.append(key)
            
        sxy = 0 # similarity score
        if simFunc.lower() == "jaccard":
            sxy = jaccard(x_jacc_dice,y_jacc_dice)
        elif simFunc.lower() == "dice":
            sxy = dice(x_jacc_dice,y_jacc_dice)
        elif simFunc.lower() == "cosine":
            sxy = cosine(x,y)
        else:
            sxy = pearson(x,y)
        
        similarity_scores[movieId] = sxy

    
    return similarity_scores
    

def contentBasedRecommendation(id,simFunc,directory):
    # id: id of movie
    # simFunc: similarity metric
    # n: number of recommendations
    # directory: the directory to load the datasets from

    # for testing: 263,61160 have the same word twice in title.
    # load datasets
    ratings_df = pd.read_csv(directory + '/ratings.csv')
    tags_df = pd.read_csv(directory + '/tags.csv')
    movies_df = pd.read_csv(directory + '/movies.csv')
    links_df = pd.read_csv(directory + '/links.csv')
    # genome_tags_df = pd.read_csv(directory + '/genome-tags.csv') # you can only load these two with the full dataset
    # genome_scores_df = pd.read_csv(directory + '/genome-scores.csv')

    print('ratings_df size: ', ratings_df.shape)
    print('tags_df size: ', tags_df.shape)
    print('movies_df size: ', movies_df.shape)
    print('links_df size: ', links_df.shape)
    # print('genome_tags_df size: ', genome_tags_df.shape)
    # print('genome_scores_df size: ', genome_scores_df.shape)
    
    
    title_token_tuples = []
    x_title = '' # title of wanted movie
    x_title_full = '' # title of wanted movie, without pre-processing, used for calculation of other movies
    stop_words = set(stopwords.words('english'))
    token_keys = [] # token keys for IDF dictionary

    # below loop creates the keys required for the dictionaries.
    for index,row in movies_df.iterrows():
        title = row['title']
        if row['movieId'] == id:
            x_title_full = title

        title = title.lower()
        title = re.sub(r" \(\d+\)", "", title) # remove parentheses with dates (eg. (1995))
        title = re.sub(r" \(.*", "", title) # remove all foreign titles in parentheses (from English) eg. turn shangai triad (Chinese title) to just shangai triad
        title = re.sub(r'[^a-zA-Z0-9\s]', '', title) # remove all remaining non-alphanumeric characters such as dots, question marks, dashes etc.
        title_tokens = title.split(' ')
        title_tokens_no_stopwords = [token for token in title_tokens if token not in stop_words]
        no_dupe_title_tokens = list(set(title_tokens_no_stopwords))
        genres = row['genres']
        genres = genres.lower()
        genre_tokens = genres.split('|')
        total_tokens_no_dupes = no_dupe_title_tokens + genre_tokens
        total_tokens_dupes = title_tokens_no_stopwords + genre_tokens
        for token in total_tokens_dupes: # this is for TF initialization
            title_token_tuple = (title,token)
            title_token_tuples.append(title_token_tuple)
        
        for token in total_tokens_no_dupes: # this is for IDF and TFIDF initialization
            token_keys.append(token)

        if row['movieId'] == id:
            x_title = title

    
    
    TF_x = {tuple: 0 for tuple in title_token_tuples} # term frequency dictionary -> key: (title,token) tuple, value: count of token in title
    IDF_x = {token: 0 for token in token_keys} # Inverse document frequency dictionary -> key: token, value: count of appearances of each token in the titles/genres column (if one appears two times in a movie we count it once)
    token_count = {token: 0 for token in token_keys} # used for IDF calculation
    TF_IDF_x = {token: 0 for token in token_keys} # TF-IDF dictionary -> key: (title,token) tuple, value: tfidf calculation score
    

    # loop used to get token frequency counts for IDF later
    for index,row in movies_df.iterrows():
        title = row['title']
        title = title.lower()
        title = re.sub(r" \(\d+\)", "", title) # remove parentheses with dates (eg. (1995))
        title = re.sub(r" \(.*", "", title) # remove all foreign titles in parentheses (from English) eg. turn shangai triad (Chinese title) to just shangai triad
        title = re.sub(r'[^a-zA-Z0-9\s]', '', title) # remove all remaining non-alphanumeric characters such as dots, question marks, dashes etc.
        title_tokens = title.split(' ')
        title_tokens_no_stopwords = [token for token in title_tokens if token not in stop_words]
        title_tokens_no_stopwords = list(set(title_tokens_no_stopwords)) # remove duplicates since we want to increment the counter if the word appears at least once but not for all instances.
        genres = row['genres']
        genres = genres.lower()
        genre_tokens = genres.split('|')
        total_tokens = title_tokens_no_stopwords + genre_tokens
        for token in total_tokens:
            token_count[token] += 1 # count how many times each token appears in all titles/genres for IDF , essentially this finds the denominator for IDF

    # Text pre-processing
    total_tokens_dupe_all = []
    total_tokens_all_no_dupes = []
    for index,row in movies_df.iterrows():
        title = row['title']
        title = title.lower()
        title = re.sub(r" \(\d+\)", "", title) # remove parentheses with dates (eg. (1995))
        title = re.sub(r" \(.*", "", title) # remove all foreign titles in parentheses (from English) eg. turn shangai triad (Chinese title) to just shangai triad
        title = re.sub(r'[^a-zA-Z0-9\s]', '', title) # remove all remaining non-alphanumeric characters such as dots, question marks, dashes etc.
        title_tokens = title.split(' ')
        title_tokens_no_stopwords = [token for token in title_tokens if token not in stop_words] # remove stopwords
        no_dupe_title_tokens = list(set(title_tokens_no_stopwords))
        total_tokens_dupe_all.extend(title_tokens_no_stopwords) 
        total_tokens_all_no_dupes.extend(no_dupe_title_tokens) # taking both tokens of title and genres as features

    # print('total',total_tokens_all_no_dupes)

    # TF and IDF calculation for movie x
    for index,row in movies_df.iterrows():
        title = row['title']
        title = title.lower()
        title = re.sub(r" \(\d+\)", "", title) # remove parentheses with dates (eg. (1995))
        title = re.sub(r" \(.*", "", title) # remove all foreign titles in parentheses (from English) eg. turn shangai triad (Chinese title) to just shangai triad
        title = re.sub(r'[^a-zA-Z0-9\s]', '', title) # remove all remaining non-alphanumeric characters such as dots, question marks, dashes etc.
        title_tokens = title.split(' ')
        title_tokens_no_stopwords = [token for token in title_tokens if token not in stop_words] # remove stopwords
        no_dupe_title_tokens = list(set(title_tokens_no_stopwords))
        genres = row['genres']
        genres = genres.lower()
        genre_tokens = genres.split('|')
        total_tokens_dupe_x = title_tokens_no_stopwords + genre_tokens 
        total_tokens_x_no_dupes = no_dupe_title_tokens + genre_tokens # taking both tokens of title and genres as features
        if title == x_title:
            for token in total_tokens_dupe_x:
                TF_x[(title,token)] += 1 # Calculate TF

            # for token in total_tokens_dupe_x: # alternative form for TF-IDF , doesn't change results
            #     TF_x[(title,token)] = TF_x[(title,token)] / len(total_tokens_dupe_x)

            # TF_x = {(title,token): total_tokens_dupe_x.count(token) for token in total_tokens_dupe_x}
            # TF_x = {key: TF_x.get(key, 0) + 1 for key in [(title, token) for token in total_tokens_dupe_x] if key in TF_x} # Optimize this
            # TF_x = Counter(total_tokens_dupe_x)
            # IDF_x = {token: math.log(len(movies_df) / token_count[token]) for token in total_tokens_x_no_dupes}
            # TF_IDF_x = {token: TF_x[(title,token)] * IDF_x[token] for token in total_tokens_x_no_dupes}

            

            for token in total_tokens_x_no_dupes:
                IDF_x[token] = math.log(len(movies_df) / token_count[token]) # Calculate IDF

            for token in total_tokens_x_no_dupes:
                TF_IDF_x[token] = TF_x[(title,token)] * IDF_x[token] # Calculate TF-IDF
            
            
            break

    x_jacc_dice = []
    for token in TF_IDF_x.keys():
        if TF_IDF_x[token] > 0:
            x_jacc_dice.append(token)


    tf_idf_x_list = []
    # for key in TF_IDF_x.keys():
    #         tf_idf_x_list.append(TF_IDF_x[key])
    tf_idf_x_list = list(TF_IDF_x.values())
    # print('tf_idf_x: ',len(TF_x))

    similarity_scores = {}

    for index,row in movies_df.iterrows():
        title = row['title']
        if title == x_title_full: # make sure that we remove the similarity score of moviex with itself
            continue
        title = title.lower()
        title = re.sub(r" \(\d+\)", "", title) # remove parentheses with dates (eg. (1995))
        title = re.sub(r" \(.*", "", title) # remove all foreign titles in parentheses (from English) eg. turn shangai triad (Chinese title) to just shangai triad
        title = re.sub(r'[^a-zA-Z0-9\s]', '', title) # remove all remaining non-alphanumeric characters such as dots, question marks, dashes etc.
        title_tokens = title.split(' ')
        title_tokens_no_stopwords = [token for token in title_tokens if token not in stop_words]
        no_dupe_title_tokens = list(set(title_tokens_no_stopwords))
        genres = row['genres']
        genres = genres.lower()
        genre_tokens = genres.split('|')
        total_tokens_y_no_dupes = no_dupe_title_tokens + genre_tokens # taking both tokens of title and genres as features
        total_tokens_y_dupes = title_tokens_no_stopwords + genre_tokens
        tf_idf_y_list = []
        TF_other_movies = {tuple: 0 for tuple in title_token_tuples} # reset all dictionaries so we can get the values of only the specific movie in the loop
        IDF_other_movies = {token: 0 for token in token_keys}
        TF_IDF_other_movies = {token: 0 for token in token_keys}  
        for token in total_tokens_y_dupes:
            TF_other_movies[(title,token)] += 1
            
        # TF_other_movies = {(title,token): total_tokens_y_dupes.count(token) for token in total_tokens_y_dupes}
        # IDF_other_movies = {token: math.log(len(movies_df) / token_count[token]) for token in total_tokens_y_no_dupes}
        # TF_IDF_other_movies = {token: TF_other_movies[(title,token)] * IDF_other_movies[token] for token in total_tokens_y_no_dupes}

        
        # for token in total_tokens_y_dupes:
        #     TF_other_movies[(title,token)] = TF_other_movies[(title,token)] / len(total_tokens_y_dupes)

        for token in total_tokens_y_no_dupes:
            IDF_other_movies[token] = math.log(len(movies_df) / token_count[token])

        for token in total_tokens_y_no_dupes:
            TF_IDF_other_movies[token] = TF_other_movies[(title,token)] * IDF_other_movies[token]

        tf_idf_y_list = list(TF_IDF_other_movies.values())
            
        
        # for key in TF_IDF_other_movies.keys():
        #     tf_idf_y_list.append(TF_IDF_other_movies[key])


        y_jacc_dice = []
        # tf_idf_y_list = normalizer(tf_idf_y_list)
        for token in TF_IDF_other_movies.keys():
            if TF_IDF_other_movies[token] > 0:
                y_jacc_dice.append(token)

        # print('tf_y',tf_idf_y_list)
        sxy = 0 # similarity score
        if simFunc.lower() == "jaccard":
            sxy = jaccard(x_jacc_dice,y_jacc_dice)
        elif simFunc.lower() == "dice":
            sxy = dice(x_jacc_dice,y_jacc_dice)
        elif simFunc.lower() == "cosine":
            sxy = cosine(tf_idf_x_list,tf_idf_y_list)
        else:
            sxy = pearson(tf_idf_x_list,tf_idf_y_list)            

        similarity_scores[row['movieId']] = sxy
    
    return similarity_scores

def hybrid(userId,movieId,simFunc,k,n,directory):
    # userId : the id of the user (for user to user)
    # movieId : the id of the movie (for any other recommendation algorithm)
    # simFunc: similarity metric
    # k: most similar users for user to user
    # n: number of recommendations

    # Logic here is that we take a user and a movie that they've seen as input
    # and we add weights to the scores of the algorithms , re-sorting them
    # and getting the best of both collaborative and content based filtering.

    scores_u2u = userToUser(userId,simFunc,k,directory)
    scores_tag = tagBasedRecommendation(movieId,simFunc,directory)
    scores_tfidf = contentBasedRecommendation(movieId,simFunc,directory)

    # sort scores in descending order
    sorted_u2u = dict(sorted(scores_u2u.items(), key=lambda item: item[1], reverse=True)) 
    sorted_tag = dict(sorted(scores_tag.items(), key=lambda item: item[1], reverse=True))
    sorted_tfidf = dict(sorted(scores_tfidf.items(), key=lambda item: item[1], reverse=True))

    # get top n values for all scores
    
    first_n_keys_u2u = list(sorted_u2u.keys())[:n]
    first_n_values_u2u = list(sorted_u2u.values())[:n]

    first_n_keys_tag = list(sorted_tag.keys())[:n]
    first_n_values_tag = list(sorted_tag.values())[:n]

    first_n_keys_tfidf = list(sorted_tfidf.keys())[:n]
    first_n_values_tfidf = list(sorted_tfidf.values())[:n]
    

    # Find common movies in the metrics
    common_movie_ids = set(first_n_keys_u2u) & set(first_n_keys_tag) & set(first_n_keys_tfidf)

    # Weights for each recommendation algorithm
    weight1 = 0.4
    weight2 = 0.5
    weight3 = 0.3

    # if there are no common movies simply sort all scores and send the best ones back
    if (len(common_movie_ids) < n):
        for val in sorted_u2u.values():
            val *= weight1

        for val in sorted_tfidf.values():
            val *= weight3
        
        for val in sorted_tag.values():
            val *= weight2

        hybrid_final_scores = {}
        hybrid_final_scores.update(sorted_u2u)
        hybrid_final_scores.update(sorted_tfidf)
        hybrid_final_scores.update(sorted_tag)
        return hybrid_final_scores

    

    hybrid_scores = {}

    # Perform linear combination on the top n scores
    for movieId in common_movie_ids:
        u2u_score = sorted_u2u(movieId,0)
        tag_score = sorted_tag(movieId,0)
        tfidf_score = sorted_tfidf(movieId,0)

        hybrid_score = weight1 * u2u_score + weight2 * tag_score + weight3 * tfidf_score
        hybrid_scores[movieId] = hybrid_score

    return hybrid_scores
# ratings_df = pd.read_csv('100_datasets/ratings.csv')
# pivot_table = ratings_df.pivot_table(index='userId', columns='movieId', values='rating',fill_value=0.5)
# pivot_table = pivot_table.T # transpose so movies are the indexes

# part_1_pt = pivot_table.iloc[:442, :] # large_pivot.iloc[:, :3]
# sim_scores = {}
# sim_scores.update(calculate_similarity_on_pivot_subset(part_1_pt,part_1_pt,"cosine"))
# print('sim_scores',sim_scores)


def main():
    arguments = sys.argv[1:]
    print(len(arguments))
    if len(arguments) < 4:
        print("ERROR: please provide all arguments.")
        print('example: python preprocess.py -d 100_datasets -a user')
        return

    algorithm = arguments[3]
    
    movies_df = pd.read_csv(arguments[1] + '/movies.csv') # load this to show recommended movies as output
    ratings_df = pd.read_csv(arguments[1] + '/ratings.csv')
    tags_df = pd.read_csv(arguments[1] + '/tags.csv')
    links_df = pd.read_csv(arguments[1] + '/links.csv')
    # genome_tags_df = pd.read_csv(arguments[1] + '/genome-tags.csv') # you can only load these two with the full dataset
    # genome_scores_df = pd.read_csv(arguments[1] + '/genome-scores.csv')

    print('ratings_df size: ', ratings_df.shape)
    print('tags_df size: ', tags_df.shape)
    print('movies_df size: ', movies_df.shape)
    print('links_df size: ', links_df.shape)
    # print('genome_tags_df size: ', genome_tags_df.size)
    # print('genome_scores_df size: ', genome_scores_df.size)

    sim_metrics = ['jaccard','dice','cosine','pearson']
    if algorithm == "user":
        user_counter = 0
        for userId in ratings_df['userId'].unique():
            for metric in sim_metrics:
                rx = userToUser(userId,metric,128,arguments[1])
                print('done for user: ', user_counter , ' metric:', metric)
                user_counter += 1
                with open('text_files/user_to_user/user_' + str(userId) + '_' + metric + '.txt', 'w') as file:
                    for key in rx.keys():
                        file.write(str(key) + ' ' + str(rx[key]) + "\n")
        
    elif algorithm == "item":
        for userId in ratings_df['userId'].unique():
            for metric in sim_metrics:
                rx = itemToItem(userId,metric,128,arguments[1])
                print('done for user: ', userId , ' metric:', metric)
                with open('text_files/item_to_item/user_' + str(userId) + '_' + metric + '.txt', 'w') as file:
                    for key in rx.keys():
                        file.write(str(key) + ' ' + str(rx[key]) + "\n")
                
        # sorted_rx = dict(sorted(rx.items(), key=lambda item: item[1], reverse=True)) # sort recommendation scores in descending order
        # first_n_keys = list(sorted_rx.keys())[:number_of_recommendations] # get top n keys
        # recommended_movies = movies_df[movies_df['movieId'].isin(first_n_keys)]
        # print("\nHere are your top", number_of_recommendations, "recommendations: \n")
        # print(recommended_movies['title'])
            
        
    # elif algorithm == "item":
    #     recommendation_scores = itemToItem(input,similarity_metric,128,arguments[1])
    #     sorted_rec_scores = dict(sorted(recommendation_scores.items(), key=lambda item: item[1], reverse=True)) # sort recommendation scores in descending order
    #     first_n_keys = list(sorted_rec_scores.keys())[:number_of_recommendations] # get top n keys
    #     recommended_movies = movies_df[movies_df['movieId'].isin(first_n_keys)]
    #     print("\nHere are your top", number_of_recommendations, "recommendations: \n")
    #     print(recommended_movies['title'])
    # elif algorithm == "tag":
    #     similarity_scores = tagBasedRecommendation(input,similarity_metric,arguments[1])
    #     sorted_sxy = dict(sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)) # sort similarity scores in descending order
    #     first_n_keys = list(sorted_sxy.keys())[:number_of_recommendations] # get top n keys
    #     recommended_movies = movies_df[movies_df['movieId'].isin(first_n_keys)]
    #     print("\nHere are your top", number_of_recommendations, "recommendations: \n")
    #     print(recommended_movies['title'])
    # elif algorithm == "title":
    #     similarity_scores = contentBasedRecommendation(input,similarity_metric,arguments[1])
    #     sorted_sxy = dict(sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)) # sort similarity scores in descending order
    #     first_n_keys = list(sorted_sxy.keys())[:number_of_recommendations] # get top n keys
    #     recommended_movies = movies_df[movies_df['movieId'].isin(first_n_keys)]
    #     print("\nHere are your top", number_of_recommendations, "recommendations: \n")
    #     print(recommended_movies['title'])
    # elif algorithm == "hybrid":
    #     if len(arguments) < 11:
    #         print("ERROR: please provide all arguments.")
    #         print('example: python recommender.py -d datasets -n 10 -s hybrid -a user -i 2 1')
    #         return
    #     user_input = int(arguments[9])
    #     item_input = int(arguments[10])
    #     hybrid_scores = hybrid(user_input,item_input,similarity_metric,128,number_of_recommendations,arguments[1])
    #     sorted_hybrid_scores = dict(sorted(hybrid_scores.items(), key=lambda item: item[1], reverse=True))
    #     first_n_keys = list(sorted_hybrid_scores.keys())[:number_of_recommendations] # get top n keys
    #     recommended_movies = movies_df[movies_df['movieId'].isin(first_n_keys)]
    #     print("\nHere are your top", number_of_recommendations, "recommendations: \n")
    #     print(recommended_movies['title'])
main()

