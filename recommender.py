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
    return (2*len(x.intersection(y))) / (len(x) + len(y)) 

def cosine(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def pearson(x,y,n):
    # x,y: sets 
    # n: sample size
    x_sum = sum(x)
    x_mean = x_sum / len(x)
    y_sum = sum(y)
    y_mean = y_sum / len(y)
    numerator = 0
    denominator = 0
    sum1 = 0
    sum2 = 0
    for i in range(0,n):
        numerator += (x[i] - x_mean) * (y[i] - y_mean)
        print(numerator)
        # denominator += math.sqrt((x[i] - x_mean) ** 2) * math.sqrt((y[i] - y_mean) ** 2)
        sum1 += math.sqrt((x[i] - x_mean) ** 2)
        sum2 += math.sqrt((y[i] - y_mean) ** 2)

    denominator = sum1 * sum2
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

# print('pearson: ',pearson(x,z,2))
user_x_ratings = ratings_df[ratings_df['userId'] == 1]
movie_ratings_y_users = pd.DataFrame({})

movie_ratings_y_users = ratings_df[(ratings_df['movieId'].isin(user_x_ratings['movieId'])) & (ratings_df['userId'] != 1)]

print(user_x_ratings.size)
print(movie_ratings_y_users['userId'].nunique())

def userToUser(id,simFunc,k):
    # id: id of the user for recommendation
    # simFunc: the similarity function used
    # k: sample of users with similar ratings to user for recommendation
    rx = []
    user_x_ratings = ratings_df[ratings_df['userId'] == id]  # get ratings of x user
    movie_ratings_y_users = ratings_df[(ratings_df['movieId'].isin(user_x_ratings['movieId'])) & (ratings_df['userId'] != id)] # get all ratings of users y for the same movies as x


    metric = 0
    if simFunc.lower() == "jaccard":
        metric = jaccard(x,y)
    elif simFunc.lower() == "dice":
        metric = dice(x,y)
    elif simFunc.lower() == "cosine":
        metric = cosine(x,y)
    else:
        metric = pearson(x,y)

    for i in range(0,len(n)):
        print(id)
