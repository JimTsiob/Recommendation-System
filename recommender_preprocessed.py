import pandas as pd # pip install pandas
import math
import sys
import re


def main():
    arguments = sys.argv[1:]
    if len(arguments) < 10:
        print("ERROR: please provide all arguments.")
        print('example: python recommender_preprocessed.py -d datasets -n 10 -s jaccard -a user -i 2')
        return

    number_of_recommendations = int(arguments[3])
    input = int(arguments[9])
    similarity_metric = arguments[5]

    movies_df = pd.read_csv(arguments[1] + '/movies.csv') # load this to show recommended movies as output

    rx = {}
    if arguments[7] == "user":
        with open('text_files/user_to_user/user_' + str(input) + '_' + similarity_metric +'.txt', 'r') as file:
            for line in file:
                key,value = map(float, line.strip().split())
                rx[key] = value
        
        sorted_rx = dict(sorted(rx.items(), key=lambda item: item[1], reverse=True)) # sort recommendation scores in descending order
        first_n_keys = list(sorted_rx.keys())[:number_of_recommendations] # get top n keys
        recommended_movies = movies_df[movies_df['movieId'].isin(first_n_keys)]
        print("\nHere are your top", number_of_recommendations, "recommendations: \n")
        print(recommended_movies['title'])

    elif arguments[7] == "item":
        with open('text_files/item_to_item/user_' + str(input) + '_' + similarity_metric +'.txt', 'r') as file:
            for line in file:
                key,value = map(float, line.strip().split())
                rx[key] = value

        sorted_rec_scores = dict(sorted(rx.items(), key=lambda item: item[1], reverse=True)) # sort recommendation scores in descending order
        first_n_keys = list(sorted_rec_scores.keys())[:number_of_recommendations] # get top n keys
        recommended_movies = movies_df[movies_df['movieId'].isin(first_n_keys)]
        print("\nHere are your top", number_of_recommendations, "recommendations: \n")
        print(recommended_movies['title'])

    elif arguments[7] == "tag":
        with open('text_files/tag_based/movie_' + str(input) + '_' + similarity_metric +'.txt', 'r') as file:
            for line in file:
                key,value = map(float, line.strip().split())
                rx[key] = value

        sorted_sxy = dict(sorted(rx.items(), key=lambda item: item[1], reverse=True)) # sort similarity scores in descending order
        first_n_keys = list(sorted_sxy.keys())[:number_of_recommendations] # get top n keys
        recommended_movies = movies_df[movies_df['movieId'].isin(first_n_keys)]
        print("\nHere are your top", number_of_recommendations, "recommendations: \n")
        print(recommended_movies['title'])

    elif arguments[7] == "title":
        with open('text_files/content_based/movie_' + str(input) + '_' + similarity_metric +'.txt', 'r') as file:
            for line in file:
                key,value = map(float, line.strip().split())
                rx[key] = value

        sorted_sxy = dict(sorted(rx.items(), key=lambda item: item[1], reverse=True)) # sort similarity scores in descending order
        first_n_keys = list(sorted_sxy.keys())[:number_of_recommendations] # get top n keys
        recommended_movies = movies_df[movies_df['movieId'].isin(first_n_keys)]
        print("\nHere are your top", number_of_recommendations, "recommendations: \n")
        print(recommended_movies['title'])

    elif arguments[7] == "hybrid":
        user_input = input
        movie_input = arguments[10]
        if len(arguments) < 11:
            print("ERROR: please provide all arguments.")
            print('example: python recommender.py -d datasets -n 10 -s hybrid -a user -i 2 1')
            return
        
        with open('text_files/hybrid/score_' + str(user_input) + '_' + str(movie_input) + '_' + similarity_metric +'.txt', 'r') as file:
            for line in file:
                key,value = map(float, line.strip().split())
                rx[key] = value

        sorted_sxy = dict(sorted(rx.items(), key=lambda item: item[1], reverse=True)) # sort similarity scores in descending order
        first_n_keys = list(sorted_sxy.keys())[:number_of_recommendations] # get top n keys
        recommended_movies = movies_df[movies_df['movieId'].isin(first_n_keys)]
        print("\nHere are your top", number_of_recommendations, "recommendations: \n")
        print(recommended_movies['title'])
        
main()