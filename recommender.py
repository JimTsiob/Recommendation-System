import pandas as pd
import numpy as np
import math


ratings_df = pd.read_csv('datasets/ratings.csv')
tags_df = pd.read_csv('datasets/tags.csv')
movies_df = pd.read_csv('datasets/movies.csv')
links_df = pd.read_csv('datasets/links.csv')
genome_tags_df = pd.read_csv('datasets/genome-tags.csv')
genome_scores_df = pd.read_csv('datasets/genome-scores.csv')

print('ratings_df size: ', ratings_df.size)
print('tags_df size: ', tags_df.size)
print('movies_df size: ', movies_df.size)
print('links_df size: ', links_df.size)
print('genome_tags_df size: ', genome_tags_df.size)
print('genome_scores_df size: ', genome_scores_df.size)

# Metrics creation
def jaccard(a,b):
    """Define Jaccard Similarity function for two sets"""
    intersection = len(list(set(a).intersection(b)))
    union = (len(a) + len(b)) - intersection
    return float(intersection) / union

def dice(x,y):
    return (2*len(list(set(x).intersection(y)))) / (len(x) + len(y)) 

def cosine(a,b):
    # if the two sets do not have the same length we take a subsample
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
        root1 = math.sqrt(sum1)
        sum2 += b[i] ** 2
        root2 = math.sqrt(sum2)

    denominator = root1 * root2 
    return numerator / denominator

def pearson(x,y):
    # x,y: sets 
    # if the two sets do not have the same length we take a subsample
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

print('pearson: ',pearson(x,y))
user_x_ratings = ratings_df[ratings_df['userId'] == 1]
movie_ratings_y_users = pd.DataFrame({})

movie_ratings_y_users = ratings_df[(ratings_df['movieId'].isin(user_x_ratings['movieId'])) & (ratings_df['userId'] != 1)]
y_users_sample = movie_ratings_y_users.sample(n=128)

print(user_x_ratings.size)
print(movie_ratings_y_users['userId'].nunique())

def userToUser(id,simFunc,k):
    # id: id of the user for recommendation
    # simFunc: the similarity function used
    # k: k users with ratings most similar to user for recommendation

    user_x_ratings = ratings_df[ratings_df['userId'] == id]  # get ratings of x user
    movie_ratings_y_users = ratings_df[(ratings_df['movieId'].isin(user_x_ratings['movieId'])) & (ratings_df['userId'] != id)] # get all ratings of users y for the same movies as x
    
    sample_y_user_ids = movie_ratings_y_users['userId'].drop_duplicates().sample(n=k) # get k users most similar to user x ratings.
    y_users_sample = movie_ratings_y_users[movie_ratings_y_users['userId'].isin(sample_y_user_ids)]

    rx = []
    x = []
    y = []
    x.append(user_x_ratings['rating']) # ratings of user x
    y.append(y_users_sample['rating']) # ratings of k users

    sxy = 0 # similarity score
    if simFunc.lower() == "jaccard":
        sxy = jaccard(x,y)
    elif simFunc.lower() == "dice":
        sxy = dice(x,y)
    elif simFunc.lower() == "cosine":
        sxy = cosine(x,y)
    else:
        sxy = pearson(x,y)
    
    # calculation of formula for recommendation score rxi
    numerator = 0
    denominator = 0
    for i in range(0,len(y)):
        numerator += sxy * y[i]
        denominator += sxy
        
    

    

    
