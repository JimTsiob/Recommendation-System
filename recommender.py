import pandas as pd
import numpy as np
import math


ratings_df = pd.read_csv('100_datasets/ratings.csv')
tags_df = pd.read_csv('100_datasets/tags.csv')
movies_df = pd.read_csv('100_datasets/movies.csv')
links_df = pd.read_csv('100_datasets/links.csv')
# genome_tags_df = pd.read_csv('100_datasets/genome-tags.csv')
# genome_scores_df = pd.read_csv('100_datasets/genome-scores.csv')

print('ratings_df size: ', ratings_df.size)
print('tags_df size: ', tags_df.size)
print('movies_df size: ', movies_df.size)
print('links_df size: ', links_df.size)
# print('genome_tags_df size: ', genome_tags_df.size)
# print('genome_scores_df size: ', genome_scores_df.size)

# Metrics creation
def jaccard(a,b):
    """Define Jaccard Similarity function for two sets"""
    intersection = len(list(set(a).intersection(b)))
    union = (len(a) + len(b)) - intersection
    return float(intersection) / union

def dice(x,y):
    return (2*len(list(set(x).intersection(y)))) / (len(x) + len(y)) 

def cosine(a,b):
    # if the two lists do not have the same length we take a subsample
    n = 0
    if len(a) > len(b):
        n = len(b)
    elif len(a) < len(b):
        n = len(a)
    else:
        n = len(a) # at this part they're both the same size so we don't care which one we'll take.
        
    numerator = 0
    denominator = 0
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
    denominator = root1 * root2
    return numerator / denominator

def pearson(x,y): 
    # if the two lists do not have the same length we take a subsample
    n = 0
    if len(x) > len(y):
        n = len(y)
    elif len(x) < len(y):
        n = len(x)
    else:
        n = len(x) # at this part they're both the same size so we don't care which one we'll take.

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
    denominator = root1 * root2
    return numerator / denominator

a = {1, 2, 3, 4}
b = {3, 4, 5, 6}
x = [1,2,3,4]
y = [3,4,5,6]
z = [1,2,3,4]
ena = [4.0]
dyo = [3.5,6.0,4.0]
# x = np.arange(10, 20)
# y = np.array([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])

# print('pearson: ',pearson(x,y))
# user_x_ratings = ratings_df[ratings_df['userId'] == 1]
# movie_ratings_y_users = pd.DataFrame({})

# movie_ratings_y_users = ratings_df[(ratings_df['movieId'].isin(user_x_ratings['movieId'])) & (ratings_df['userId'] != 1)]
# # y_users_sample = movie_ratings_y_users.sample(n=128)
# x = []
# for rating in user_x_ratings['rating']:
#     x.append(rating)

def normalizer(x):
    # x: Iterable (most likely list)
    # normalizer to transform the ratings in the interval [0,1]
    norm_x = []
    for num in x:
        if max(x) == min(x): # this is to avoid divide by zero error, could happen with some arrays
            norm_num = 0.1
        else:
            norm_num = (num - min(x)) / (max(x) - min(x))
        norm_x.append(norm_num)
    
    return norm_x


def userToUser(id,simFunc,k,n):
    # id: id of the user for recommendation
    # simFunc: the similarity function used
    # k: k users with ratings most similar to user for recommendation
    # n: return top n recommendations

    user_x_ratings = ratings_df[ratings_df['userId'] == id]  # get ratings of x user
    user_x_movieIds = user_x_ratings['movieId'].unique() # get movie ids of user x to filter them out later on
    movie_ratings_y_users = ratings_df[(ratings_df['movieId'].isin(user_x_ratings['movieId'])) & (ratings_df['userId'] != id)] # get all ratings of users y for the same movies as x
    

    rx = {} # dictionary to store recommendation scores key: movieId , value: score
    x = []

    for rating in user_x_ratings['rating']: # ratings of user x
        x.append(rating)
     

    # find similarity scores for all users y with user x
    similarity_scores = {} # dictionary to hold sim scores , key: userId value: similarity score
    for userId in movie_ratings_y_users['userId']:
        y_similar_movie_rating = movie_ratings_y_users[movie_ratings_y_users['userId'] == userId] # get ratings for each user y for the same movies that x has watched
        y = []
        for rating in y_similar_movie_rating['rating']:
            y.append(rating)

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
    first_k_users = list(sorted_sxy.keys())[:k] # get k most similar users
    y_user_ratings_different_movies_from_x = ratings_df[(ratings_df['userId'].isin(first_k_users)) & (~ratings_df['movieId'].isin(user_x_movieIds))] # get ratings of k most similar users y to x for other movies that x has not watched.
    
    # calculation of formula for recommendation score rxi for each movie not watched by user x
    for movieId in y_user_ratings_different_movies_from_x['movieId']:
        y_rating = y_user_ratings_different_movies_from_x[y_user_ratings_different_movies_from_x['movieId'] == movieId] # get ratings for each movie
        y = []
        for rating in y_rating['rating']:
            y.append(rating)

        # normalization
        y = normalizer(y)

        numerator = 0
        denominator = 0
        for i in range(0,len(y)):
            numerator += sxy * y[i]
            denominator += sxy

        rxi = numerator / denominator
        rx[movieId] = rxi
    
    sorted_rx = dict(sorted(rx.items(), key=lambda item: item[1], reverse=True)) # sort recommendation scores in descending order
    first_n_keys = list(sorted_rx.keys())[:n] # get top n keys
    recommended_movies = movies_df[movies_df['movieId'].isin(first_n_keys)]
    print("\nHere are your top", n, "recommendations: \n")
    print(recommended_movies['title'])
        
    
        
# testing
userToUser(1,"cosine",128,10)

    

    
