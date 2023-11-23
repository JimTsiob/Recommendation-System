import pandas as pd # pip install pandas
import numpy as np # pip install numpy
import math
import sys
import nltk # pip install nltk
# nltk.download('stopwords') # <- run this if you don't have the stopwords already on your machine.
from nltk.corpus import stopwords
import re




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
    norm_x = []
    epsilon = 1e-9 # this is to avoid divide by zero error, in case of same size same value vectors.
    for num in x:
        if max(x) == min(x): # this is to avoid divide by zero error, could happen with some arrays
            norm_num = (num - min(x)) / (max(x) - min(x) + epsilon)
        else:
            norm_num = (num - min(x)) / (max(x) - min(x))
        norm_x.append(norm_num)
    
    return norm_x

def normalizeNum(x,min,max): # used for each separate rating in recommendation score calculation
    num = 0
    num = (x-min) / (max - min)
    return num

def calculate_similarity_for_pivot_1_to_10(pt1,pt2,pt3,pt4,pt5,pt6,pt7,pt8,pt9,pt10,simFunc):
    sim_score_dict = {}
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt1,pt1,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt1,pt2,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt1,pt3,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt1,pt4,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt1,pt5,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt1,pt6,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt1,pt7,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt1,pt8,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt1,pt9,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt1,pt10,simFunc))

    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt2,pt1,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt2,pt2,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt2,pt3,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt2,pt4,simFunc))    
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt2,pt5,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt2,pt6,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt2,pt7,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt2,pt8,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt2,pt9,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt2,pt10,simFunc))

    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt3,pt1,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt3,pt2,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt3,pt3,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt3,pt4,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt3,pt5,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt3,pt6,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt3,pt7,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt3,pt8,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt3,pt9,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt3,pt10,simFunc))

    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt4,pt1,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt4,pt2,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt4,pt3,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt4,pt4,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt4,pt5,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt4,pt6,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt4,pt7,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt4,pt8,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt4,pt9,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt4,pt10,simFunc))

    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt5,pt1,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt5,pt2,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt5,pt3,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt5,pt4,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt5,pt5,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt5,pt6,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt5,pt7,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt5,pt8,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt5,pt9,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt5,pt10,simFunc))

    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt6,pt1,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt6,pt2,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt6,pt3,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt6,pt4,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt6,pt5,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt6,pt6,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt6,pt7,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt6,pt8,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt6,pt9,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt6,pt10,simFunc))

    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt7,pt1,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt7,pt2,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt7,pt3,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt7,pt4,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt7,pt5,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt7,pt6,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt7,pt7,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt7,pt8,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt7,pt9,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt7,pt10,simFunc))

    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt8,pt1,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt8,pt2,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt8,pt3,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt8,pt4,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt8,pt5,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt8,pt6,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt8,pt7,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt8,pt8,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt8,pt9,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt8,pt10,simFunc))

    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt9,pt1,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt9,pt2,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt9,pt3,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt9,pt4,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt9,pt5,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt9,pt6,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt9,pt7,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt9,pt8,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt9,pt9,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt9,pt10,simFunc))

    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt10,pt1,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt10,pt2,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt10,pt3,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt10,pt4,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt10,pt5,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt10,pt6,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt10,pt7,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt10,pt8,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt10,pt9,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt10,pt10,simFunc))

    return sim_score_dict



def calculate_subset_pivots_from_11_to_17(pt11,pt12,pt13,pt14,pt15,pt16,pt17,simFunc):
    sim_score_dict = {}
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt11,pt11,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt11,pt12,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt11,pt13,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt11,pt14,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt11,pt15,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt11,pt16,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt11,pt17,simFunc))

    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt12,pt11,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt12,pt12,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt12,pt13,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt12,pt14,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt12,pt15,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt12,pt16,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt12,pt17,simFunc))

    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt13,pt11,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt13,pt12,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt13,pt13,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt13,pt14,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt13,pt15,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt13,pt16,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt13,pt17,simFunc))

    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt14,pt11,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt14,pt12,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt14,pt13,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt14,pt14,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt14,pt15,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt14,pt16,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt14,pt17,simFunc))

    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt15,pt11,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt15,pt12,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt15,pt13,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt15,pt14,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt15,pt15,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt15,pt16,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt15,pt17,simFunc))

    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt16,pt11,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt16,pt12,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt16,pt13,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt16,pt14,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt16,pt15,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt16,pt16,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt16,pt17,simFunc))

    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt17,pt11,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt17,pt12,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt17,pt13,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt17,pt14,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt17,pt15,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt17,pt16,simFunc))
    sim_score_dict.update(calculate_similarity_on_pivot_subset(pt17,pt17,simFunc))

    return sim_score_dict



def userToUser(id,simFunc,k,n,directory):
    # id: id of the user for recommendation
    # simFunc: the similarity function used
    # k: k users with ratings most similar to user for recommendation
    # n: return top n recommendations

    # load datasets
    ratings_df = pd.read_csv(directory + '/ratings.csv')
    tags_df = pd.read_csv(directory + '/tags.csv')
    movies_df = pd.read_csv(directory + '/movies.csv')
    links_df = pd.read_csv(directory + '/links.csv')
    # genome_tags_df = pd.read_csv(arguments[1] + '/genome-tags.csv') # you can only load these two with the full dataset
    # genome_scores_df = pd.read_csv(arguments[1] + '/genome-scores.csv')

    print('ratings_df size: ', ratings_df.size)
    print('tags_df size: ', tags_df.size)
    print('movies_df size: ', movies_df.size)
    print('links_df size: ', links_df.size)
    # print('genome_tags_df size: ', genome_tags_df.size)
    # print('genome_scores_df size: ', genome_scores_df.size)

    user_x_ratings = ratings_df[ratings_df['userId'] == id]  # get ratings of x user
    user_x_movieIds = user_x_ratings['movieId'].unique() # get movie ids of user x to filter them out later on
    movie_ratings_y_users = ratings_df[(ratings_df['movieId'].isin(user_x_ratings['movieId'])) & (ratings_df['userId'] != id)] # get all ratings of users y for the same movies as x
    

    rx = {} # dictionary to store recommendation scores key: movieId , value: score
    
    x_rating_dict = dict(zip(user_x_ratings['movieId'], user_x_ratings['rating']))
     

    # find similarity scores for all users y with user x
    similarity_scores = {} # dictionary to hold sim scores , key: userId value: similarity score
    for userId in movie_ratings_y_users['userId']:
        y_similar_movie_rating = movie_ratings_y_users[movie_ratings_y_users['userId'] == userId] # get ratings for each user y for the same movies that x has watched

        x = []
        y = [] # these two lists will hold the ratings of user x and user y.
        # the dictionary below is used to keep only the ratings of the movies that both x and y have watched.
        y_rating_dict = dict(zip(y_similar_movie_rating['movieId'], y_similar_movie_rating['rating'])) # y rating dictionary, key: movieId , value: rating
        common_ratings = set(x_rating_dict.keys()).intersection(y_rating_dict.keys())
        common_ratings_dict = {key: (x_rating_dict[key], y_rating_dict[key]) for key in common_ratings}
        for val in common_ratings_dict.values():
            x.append(val[0])
            y.append(val[1])

        sxy = 0 # similarity score
        if simFunc.lower() == "jaccard":
            x = normalizer(x)
            y = normalizer(y)
            sxy = jaccard(x,y)
        elif simFunc.lower() == "dice":
            x = normalizer(x)
            y = normalizer(y)
            sxy = dice(x,y)
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
    
    sorted_rx = dict(sorted(rx.items(), key=lambda item: item[1], reverse=True)) # sort recommendation scores in descending order
    first_n_keys = list(sorted_rx.keys())[:n] # get top n keys
    recommended_movies = movies_df[movies_df['movieId'].isin(first_n_keys)]
    print("\nHere are your top", n, "recommendations: \n")
    print(recommended_movies['title'])
        
    

def itemToItem(id,simFunc,k,n,directory):
    # logic here is to find all similarity scores for pairs of movies that user x has watched
    # and movies that user x hasn't watched. Get the k most similar pairs (highest similarity score) 
    # then predict the ratings of user x for the other movies
    # (calculate recommendation score essentially) and print out the top n recommendations.

    # load datasets
    ratings_df = pd.read_csv(directory + '/ratings.csv')
    tags_df = pd.read_csv(directory + '/tags.csv')
    movies_df = pd.read_csv(directory + '/movies.csv')
    links_df = pd.read_csv(directory + '/links.csv')
    # genome_tags_df = pd.read_csv(arguments[1] + '/genome-tags.csv') # you can only load these two with the full dataset
    # genome_scores_df = pd.read_csv(arguments[1] + '/genome-scores.csv')

    print('ratings_df size: ', ratings_df.size)
    print('tags_df size: ', tags_df.size)
    print('movies_df size: ', movies_df.size)
    print('links_df size: ', links_df.size)
    # print('genome_tags_df size: ', genome_tags_df.size)
    # print('genome_scores_df size: ', genome_scores_df.size)

    user_x_ratings = ratings_df[ratings_df['userId'] == id] # get ratings of user x
    user_x_movieIds = user_x_ratings['movieId'].unique() # get movie ids of user x to filter them out later on
    x_movie_ratings_all_users = ratings_df[ratings_df['movieId'].isin(user_x_ratings['movieId'])] # get all ratings of users y for the same movies as x
    other_movie_ratings_than_x = ratings_df[~ratings_df['movieId'].isin(user_x_ratings['movieId'])] # get all ratings of movies that x hasn't watched.
    
    # separate main dataset into smaller pivot tables in order to compute similarity metrics for all pairs of movies
    pivot_table = ratings_df.pivot_table(index='userId', columns='movieId', values='rating',fill_value=0)
    pivot_table = pivot_table.T # transpose so movies are the indexes

    # mean-centering
    # mean_centered_pivot = pivot_table.sub(pivot_table.mean(axis=1), axis=0)

    part_1_pt = pivot_table.iloc[:572, :] # large_pivot.iloc[:, :3]
    part_2_pt = pivot_table.iloc[572:1144, :]
    part_3_pt = pivot_table.iloc[1144:1716, :]
    part_4_pt = pivot_table.iloc[1716:2288, :]
    part_5_pt = pivot_table.iloc[2288:2860, :]
    part_6_pt = pivot_table.iloc[2860:3432, :]
    part_7_pt = pivot_table.iloc[3432:4004, :]
    part_8_pt = pivot_table.iloc[4004:4576, :]
    part_9_pt = pivot_table.iloc[4576:5148, :]
    part_10_pt = pivot_table.iloc[5148:5720, :]
    part_11_pt = pivot_table.iloc[5720:6292, :]
    part_12_pt = pivot_table.iloc[6292:6864, :]
    part_13_pt = pivot_table.iloc[6864:7436, :]
    part_14_pt = pivot_table.iloc[7436:8008, :]
    part_15_pt = pivot_table.iloc[8008:8580, :]
    part_16_pt = pivot_table.iloc[8580:9152, :]
    part_17_pt = pivot_table.iloc[9152:9724, :]

    print('reached before sim_score calc')
    similarity_scores = {}
    similarity_scores.update(calculate_similarity_for_pivot_1_to_10(part_1_pt,part_2_pt,part_3_pt,part_4_pt,part_5_pt,part_6_pt,part_7_pt,part_8_pt,part_9_pt,part_10_pt,simFunc))
    print('done for p1 to p10.')
    similarity_scores.update(calculate_subset_pivots_from_11_to_17(part_11_pt,part_12_pt,part_13_pt,part_14_pt,part_15_pt,part_16_pt,part_17_pt,simFunc))
    print('done for p11 to p17.')

    x_similarity_scores = {} # get only the similarity scores related with movies user x has watched
    for key in similarity_scores.keys():
        if key[0] == key[1]:
            continue # remove all cases of the same movie sim_scores (eg (121,121) = 1)

        if (key[0] in user_x_movieIds) or (key[1] in user_x_movieIds):
            x_similarity_scores[key] = similarity_scores[key]
            

    sorted_sxy = dict(sorted(x_similarity_scores.items(), key=lambda item: item[1], reverse=True)) # sort similarity scores in descending order
    k_movies_similarity_score_dict = {key: sorted_sxy[key] for key in list(sorted_sxy)[:k]} # get k most similar movies
    most_similar_movie_Ids = [] # get k most similar movie Ids to the movies user x has watched so we can check each of them later on
    for key in k_movies_similarity_score_dict.keys():
        if key[0] in user_x_movieIds:
            most_similar_movie_Ids.append(key[1])
        elif key[1] in user_x_movieIds:
            most_similar_movie_Ids.append(key[0])

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
            numerator += x_rating * k_movies_similarity_score_dict[(x_movieId,movieId)] # rating of movie x has watched * similarity_score(movie x has watched,movie to be predicted)
            denominator += k_movies_similarity_score_dict[(x_movieId,movieId)] # sum of similarity scores of movies that x has watched with the movie to be predicted
        rxi = numerator / denominator
        recommendation_scores[movieId] = rxi

    sorted_rec_scores = dict(sorted(recommendation_scores.items(), key=lambda item: item[1], reverse=True)) # sort recommendation scores in descending order
    first_n_keys = list(sorted_rec_scores.keys())[:n] # get top n keys
    recommended_movies = movies_df[movies_df['movieId'].isin(first_n_keys)]
    print("\nHere are your top", n, "recommendations: \n")
    print(recommended_movies['title'])

        
def calculate_similarity_on_pivot_subset(pivot_1,pivot_2, simFunc):
    # function used for calculating the item-to-item similarity in an optimized way.
    # Using the full dataset for this is simply impossible cost-wise as it takes a lot of time
    # even with the 100k dataset.
    similarity_scores = {}
    for movie1 in pivot_1.index:
        movie1_ratings = pivot_1.loc[movie1]
        x = movie1_ratings.tolist()
        for movie2 in pivot_2.index:
            movie2_ratings = pivot_2.loc[movie2]
            y = movie2_ratings.tolist()
            if movie2 == 1179 and movie1 == 1140:
                print('x',x)
                print('y',y)
            sxy = 0 # similarity score
            if simFunc.lower() == "jaccard":
                sxy = jaccard(x,y)
            elif simFunc.lower() == "dice":
                sxy = dice(x,y)
            elif simFunc.lower() == "cosine":
                x = normalizer(x)
                y = normalizer(y)
                sxy = cosine(x,y)
            else:
                # pearson handles normalization on it's own
                sxy = pearson(x,y)
            similarity_scores[(movie1,movie2)] = sxy
        print('iteration for movie',movie1,'is done.')
    return similarity_scores



def tagBasedRecommendation(id,simFunc,n,directory):
    # id: id of movie
    # simFunc: similarity metric
    # n: number of recommendations
    # directory: the directory to load the datasets from

    # load datasets
    ratings_df = pd.read_csv(directory + '/ratings.csv')
    tags_df = pd.read_csv(directory + '/tags.csv')
    movies_df = pd.read_csv(directory + '/movies.csv')
    links_df = pd.read_csv(directory + '/links.csv')
    # genome_tags_df = pd.read_csv(arguments[1] + '/genome-tags.csv') # you can only load these two with the full dataset
    # genome_scores_df = pd.read_csv(arguments[1] + '/genome-scores.csv')

    print('ratings_df size: ', ratings_df.shape)
    print('tags_df size: ', tags_df.shape)
    print('movies_df size: ', movies_df.shape)
    print('links_df size: ', links_df.shape)
    # print('genome_tags_df size: ', genome_tags_df.shape)
    # print('genome_scores_df size: ', genome_scores_df.shape)
    # movies to check: 60756 -> comedy, 4343, 2, 31658

    
    tags = []
    for tag in tags_df['tag']:
        tags.append(tag)

    tag_count_keys = []
    for index,row in tags_df.iterrows():
        movie = row['movieId']
        tag = row['tag']
        tag_count_keys.append((movie,tag)) # get movie tag tuple to add as key on dictionary later on

    tags_count_dict = {key: 0 for key in tag_count_keys} # dictionary that will hold tags of all movies. key: movieId value: tag
    

    for index,row in tags_df.iterrows():
        movie = row['movieId']
        tag = row['tag']
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

    sorted_sxy = dict(sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)) # sort similarity scores in descending order
    first_n_keys = list(sorted_sxy.keys())[:n] # get top n keys
    recommended_movies = movies_df[movies_df['movieId'].isin(first_n_keys)]
    print("\nHere are your top", n, "recommendations: \n")
    print(recommended_movies['title'])

def contentBasedRecommendation(id,simFunc,n,directory):
    # id: id of movie
    # simFunc: similarity metric
    # n: number of recommendations
    # directory: the directory to load the datasets from

    # for testing: 263 has the same word twice in title.
    # load datasets
    ratings_df = pd.read_csv(directory + '/ratings.csv')
    tags_df = pd.read_csv(directory + '/tags.csv')
    movies_df = pd.read_csv(directory + '/movies.csv')
    links_df = pd.read_csv(directory + '/links.csv')
    # genome_tags_df = pd.read_csv(arguments[1] + '/genome-tags.csv') # you can only load these two with the full dataset
    # genome_scores_df = pd.read_csv(arguments[1] + '/genome-scores.csv')

    print('ratings_df size: ', ratings_df.shape)
    print('tags_df size: ', tags_df.shape)
    print('movies_df size: ', movies_df.shape)
    print('links_df size: ', links_df.shape)
    # print('genome_tags_df size: ', genome_tags_df.shape)
    # print('genome_scores_df size: ', genome_scores_df.shape)
    
    
    title_token_tuples = []
    x_title = ''
    TF = {} # term frequency dictionary: key: (title,token) tuple, value: count of token in title
    stop_words = set(stopwords.words('english'))

    # below loop creates the keys required for TF.
    for index,row in movies_df.iterrows():
        title = row['title']
        title = title.lower()
        title = re.sub(r" \(\d+\)", "", title) # remove parentheses with dates (eg. (1995))
        title = re.sub(r" \(.*", "", title) # remove all foreign titles in parentheses (from English) eg. turn shangai triad (Chinese title) to just shangai triad
        title_tokens = title.split(' ')
        title_tokens_no_stopwords = [token for token in title_tokens if token not in stop_words]
        for token in title_tokens_no_stopwords:
            title_token_tuple = (title,token)
            title_token_tuples.append(title_token_tuple)

        if row['movieId'] == id:
            x_title = title

    # TF and IDF calculation for movie x
    TF_other_movies = {tuple: 0 for tuple in title_token_tuples}
    TF_x = {tuple: 0 for tuple in title_token_tuples}
    TF_x_list = [] # list to hold the counts for similarity calculation
    for index,row in movies_df.iterrows():
        title = row['title']
        title = title.lower()
        title = re.sub(r" \(\d+\)", "", title) # remove parentheses with dates (eg. (1995))
        title = re.sub(r" \(.*", "", title) # remove all foreign titles in parentheses (from English) eg. turn shangai triad (Chinese title) to just shangai triad
        title_tokens = title.split(' ')
        title_tokens_no_stopwords = [token for token in title_tokens if token not in stop_words]
        if title == x_title:
            for token in title_tokens_no_stopwords:
                TF_x[(title,token)] += 1
            for value in TF_x.values():
                TF_x_list.append(value)
            # idf below
            
            break
            
    # TF IDF for other movies, sim score calculation
    TF_other_movie_list = []
    for index,row in movies_df.iterrows():
        title = row['title']
        if title == x_title:
            continue
        title = title.lower()
        title = re.sub(r" \(\d+\)", "", title)
        title = re.sub(r" \(.*", "", title)
        title_tokens = title.split(' ')
        title_tokens_no_stopwords = [token for token in title_tokens if token not in stop_words]
        for token in title_tokens_no_stopwords:
            TF_other_movies[(title,token)] += 1
        for value in TF_other_movies.values():
            TF_other_movie_list.append(value)
        
        

    # TF_x = []
    # for key in TF.keys(): # get TF for wanted movie (we'll call it movie x from now on)
    #     if key[0] == x_title:
    #         TF_x.append(TF[key])

    # keywords = {title: 0 for title in titles} # dict of keywords , key: keyword , value: count of keyword
    # for value in TF_x.values():
    #     TF_x_list.append(value)
    # TF_x_list.append(TF_x.values())
    print('list: ',TF_x_list)
    # print('title:',x_title)
    # print('len:',len(movies_df['title']))

    return

# ratings_df = pd.read_csv('100_datasets/ratings.csv')
# pivot_table = ratings_df.pivot_table(index='userId', columns='movieId', values='rating',fill_value=0.5)
# pivot_table = pivot_table.T # transpose so movies are the indexes

# part_1_pt = pivot_table.iloc[:442, :] # large_pivot.iloc[:, :3]
# sim_scores = {}
# sim_scores.update(calculate_similarity_on_pivot_subset(part_1_pt,part_1_pt,"cosine"))
# print('sim_scores',sim_scores)


def main():
    arguments = sys.argv[1:]
    if len(arguments) < 10:
        print("ERROR: please provide all arguments.")
        print('example: python recommender.py -d datasets -n 10 -s jaccard -a user -i 2')
        return

    number_of_recommendations = int(arguments[3])
    input = int(arguments[9])
    similarity_metric = arguments[5]

    if arguments[7] == "user":
        userToUser(input,similarity_metric,128,number_of_recommendations,arguments[1])
    elif arguments[7] == "item":
        itemToItem(input,similarity_metric,128,number_of_recommendations,arguments[1])
    elif arguments[7] == "tag":
        tagBasedRecommendation(input,similarity_metric,number_of_recommendations,arguments[1])
    elif arguments[7] == "title":
        contentBasedRecommendation(input,similarity_metric,number_of_recommendations,arguments[1])

main()

