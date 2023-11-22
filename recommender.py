import pandas as pd
import numpy as np
import math
import sys





# Metrics creation
def jaccard(a,b):
    """Define Jaccard Similarity function for two sets"""
    a = [0 if element <= 2.5 else 1 for element in a]
    b = [0 if element <= 2.5 else 1 for element in b] # make vectors binary
    intersection = len(list(set(a).intersection(b)))
    union = (len(a) + len(b)) - intersection
    return float(intersection) / union

def dice(x,y):
    x = [0 if element <= 2.5 else 1 for element in x]
    y = [0 if element <= 2.5 else 1 for element in y] # make vectors binary
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

    # mean-centering
    movie_rating_dict = {} # hold all ratings for each movie 
    for movieId in ratings_df['movieId']:
        ratings = ratings_df[ratings_df['movieId'] == movieId]
        a = []
        for rating in ratings['rating']:
            a.append(rating)
        
        movie_rating_dict[movieId] = a

    means_dict = {} # hold means for each movie , key: movieId value: mean of movie's ratings
    for key in movie_rating_dict.keys():
        mean_of_movie = sum(movie_rating_dict[key]) / len(movie_rating_dict[key])
        means_dict[key] = mean_of_movie
    
    for index,row in ratings_df.iterrows():
        row['rating'] -= mean_of_movie[row['movieId']]

    user_x_ratings = ratings_df[ratings_df['userId'] == id] # get ratings of user x
    user_x_movieIds = user_x_ratings['movieId'].unique() # get movie ids of user x to filter them out later on
    x_movie_ratings_all_users = ratings_df[ratings_df['movieId'].isin(user_x_ratings['movieId'])] # get all ratings of users y for the same movies as x
    other_movie_ratings_than_x = ratings_df[~ratings_df['movieId'].isin(user_x_ratings['movieId'])] # get all ratings of movies that x hasn't watched.
    
    for movieId1 in x_movie_ratings_all_users['movieId']:
        for movieId2 in ratings_df['movieId']:
            if movieId1 == movieId2:
                continue

    # find similarities with all pairs for the items of userx
    all_other_movie_ratings_other_than_x = ratings_df[ratings_df['userId'] != id]

    x_movie_rating_dict = {} # dict that hold movie IDs and their ratings in a list. (key: movieId, value: a list with the movieIds ratings)
    for movieId in user_x_ratings['movieId']: 
        rating_list = []
        movie_ratings = ratings_df[ratings_df['movieId'] == movieId]
        for rating in movie_ratings['rating']:
            rating_list.append(rating)
        x_movie_rating_dict[movieId] = rating_list

    other_than_x_movie_ratings_dict = {} # dict that holds movieIDs (other than the ones X has watched) and their ratings
    for movieId in all_other_movie_ratings_other_than_x['movieId']:
        rating_list_y = []
        movie_ratings_y = ratings_df[ratings_df['movieId'] == movieId]
        for rating in movie_ratings_y['rating']:
            rating_list_y.append(rating)
        other_than_x_movie_ratings_dict[movieId] = rating_list_y

    # calculate similarity score of pairs for every movie of user x with all other movies that x hasn't watched
    similarity_scores = {} # dict to contain the pair similarity scores
    for x_movie_id in x_movie_rating_dict.keys():
        x = []
        x = x_movie_rating_dict[x_movie_id]
        for other_than_x_movie_id in other_than_x_movie_ratings_dict.keys():
            y = []
            y = other_than_x_movie_ratings_dict[other_than_x_movie_id]
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
            similarity_scores[(x_movie_id,other_than_x_movie_id)] = sxy

    sorted_sxy = dict(sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)) # sort similarity scores in descending order
    k_movies_similarity_score_dict = {key: sorted_sxy[key] for key in list(sorted_sxy)[:k]} # get k most similar movies
    most_similar_movie_Ids = [] # get k most similar movie Ids to the movies user x has watched
    for key in k_movies_similarity_score_dict.keys():
        most_similar_movie_Ids.append(key[1])

    most_similar_movie_df = ratings_df[ratings_df['movieId'].isin(most_similar_movie_Ids)] # filter out all other movies apart from the k most similar

    recommendation_scores = {} # recommendation score dictionary , Key: movieId Value: recommendation score
    for movieId in most_similar_movie_df['movieId']:
        # find recommendation scores for all movies that x hasn't watched.
        numerator = 0
        denominator = 0
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
    # x vector = movie 1 ratings, y vector = movie2 ratings
    # get ratings , put them in x,y lists , sim scores etc , similar procedure as above
    # get k most similar movies 
    # find recommendation scores for the k movies which user x hasn't rated.

    # recommendation score calculation (example movie with id 5 which userx hasn't watched): r(userx,5) = rating of userx for item that he has watched * similarity score(movie that x has watched, 5)

        
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

# main()
fake_df_1 = {'userId': [1, 2, 3, 4, 5],
        'movieId': ['A', 'B', 'C', 'B', 'C'],
        'rating': [4.5, 6.7, 8.2, 5.0, 8.0]}
fake_df_1 = pd.DataFrame(fake_df_1)
fake_df_2 = {'userId': [1, 4 ,5],
        'movieId': ['B', 'C', 'D'],
        'rating': [6.8, 2.4, 8.2]}
fake_df_2 = pd.DataFrame(fake_df_2)
merged_df = pd.merge(fake_df_1, fake_df_2, on='userId', how='inner')
print('merge: ',merged_df)


# columns = ['userId', 'movieId', 'rating', 'timestamp']
# ratings_df = pd.read_csv('ml-100k/u.data', sep='\t', names=columns)
# movies_df = 
# print(ratings_df.head())

ratings_df = pd.read_csv('100_datasets/ratings.csv')
tags_df = pd.read_csv('100_datasets/tags.csv')
movies_df = pd.read_csv('100_datasets/movies.csv')
links_df = pd.read_csv('100_datasets/links.csv')

user_x_ratings = ratings_df[ratings_df['userId'] == 1] # get ratings of user x
user_x_movieIds = user_x_ratings['movieId'].unique() # get movie ids of user x to filter them out later on
x_movie_ratings_all_users = ratings_df[ratings_df['movieId'].isin(user_x_ratings['movieId'])] # get all ratings of users for the same movies as x
other_movie_ratings_than_x = ratings_df[~ratings_df['movieId'].isin(user_x_ratings['movieId'])] # get all ratings of movies that x hasn't watched.

rating_dict = {}
# for index,row in ratings_df.iterrows():
#     user_id = row['userId']
#     movie_id = row['movieId']
#     rating_num = row['rating']
#     rating_dict[(user_id,movie_id)] = rating_num

pivot_table = ratings_df.pivot_table(index='userId', columns='movieId', values='rating',fill_value=0)
x_pivot_table = pivot_table
pivot_table = pivot_table.T # transpose so movies are the indexes
pivot_table = pivot_table.fillna(0) # fill all NaN values with 0
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(pivot_table)
print(sim_matrix)
# for movie1 in pivot_table.index:
#     for movie2 in pivot_table.index:
#         movie1_ratings = pivot_table.loc[movie1]
#         movie2_ratings = pivot_table.loc[movie2]

print('pivot * pivot: ', len(pivot_table) * len(pivot_table))

print(len(x_movie_ratings_all_users))
print(len(other_movie_ratings_than_x))

print(fake_df_1)
print(fake_df_2)