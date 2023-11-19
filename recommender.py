import pandas as pd
import numpy as np
import math
import sys





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
# x = np.arange(10, 20)
# y = np.array([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])

# print('pearson: ',pearson(x,y))
# user_x_ratings = ratings_df[ratings_df['userId'] == 1]
# movie_rating_dict = dict(zip(user_x_ratings['movieId'], user_x_ratings['rating']))
# print('dict: ',len(movie_rating_dict))
# # movie_ratings_y_users = pd.DataFrame({})

# movie_ratings_y_users = ratings_df[(ratings_df['movieId'].isin(user_x_ratings['movieId'])) & (ratings_df['userId'] != 1)]
# movie_rating_y_2 = ratings_df[(ratings_df['movieId'].isin(user_x_ratings['movieId'])) & (ratings_df['userId'] == 2)]
# movie_rating_dict_y2 = dict(zip(movie_rating_y_2['movieId'], movie_rating_y_2['rating']))
# print('dict_y2: ',movie_rating_dict_y2)

# common_ratings = set(movie_rating_dict.keys()).intersection(movie_rating_dict_y2.keys())
# common_ratings_dict = {key: (movie_rating_dict[key], movie_rating_dict_y2[key]) for key in common_ratings}
# common_ratings_asc_order = dict(sorted(common_ratings_dict.items()))
# # print('common: ',common_ratings_dict)
# print('common asc: ',common_ratings_asc_order)
# for val in common_ratings_dict.values():
#     print(val[1])

# new_dict = dict(zip(movie_rating_y_2['movieId'],[0.0] * len(movie_rating_y_2['movieId'])))
# print('new: ',new_dict)
# -----------------------------------------------------------------------------------------------------------

# # y_users_sample = movie_ratings_y_users.sample(n=128)
# x = []
# for rating in user_x_ratings['rating']:
#     x.append(rating)

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
    # genome_tags_df = pd.read_csv(arguments[1] + '/genome-tags.csv')
    # genome_scores_df = pd.read_csv(arguments[1] + '/genome-scores.csv')

    print('ratings_df size: ', ratings_df.size)
    print('tags_df size: ', tags_df.size)
    print('movies_df size: ', movies_df.size)
    print('links_df size: ', links_df.size)

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
        
    

def itemToItem():
    return

        
def main():
    script_name = sys.argv[0]
    arguments = sys.argv[1:]
    # Print args
    print("Arguments:", arguments)
    if len(arguments) < 10:
        print("ERROR: please provide all arguments.")
        print('example: python recommender.py -d datasets -n 10 -s jaccard -a user -i 2')
        return

    number_of_recommendations = int(arguments[3])
    input = int(arguments[9])
    similarity_metric = arguments[5]

    if arguments[7] == "user":
        userToUser(input,similarity_metric,128,number_of_recommendations,arguments[1])
    # print('genome_tags_df size: ', genome_tags_df.size)
    # print('genome_scores_df size: ', genome_scores_df.size)

main()

    

    
