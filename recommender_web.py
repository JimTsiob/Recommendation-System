from flask import Flask, render_template, request
import sys
import pandas as pd 


app = Flask(__name__)

METRICS = [
    "jaccard",
    "dice",
    "cosine",
    "pearson"
]

ALGORITHMS = [
    "user to user",
    "item to item",
    "tag based",
    "content based",
    "hybrid"
]

@app.route('/')
def index():
    return render_template('index.html', metrics=METRICS, algorithms=ALGORITHMS) # main page

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form['user_input'] 
    algorithm = request.form['algorithm']
    metric = request.form['metric']

    arguments = sys.argv[1:]
    if len(arguments) < 2:
        print("ERROR: please provide all arguments.")
        print('example: python recommender_web.py -d datasets')
        return
    
    rx = {}
    movies_df = pd.read_csv(arguments[1] + '/movies.csv') # load this to show recommended movies as output
    
    if algorithm == "user to user":
         with open('text_files/user_to_user/user_' + str(user_input) + '_' + metric +'.txt', 'r') as file:
            for line in file:
                key,value = map(float, line.strip().split())
                rx[key] = value

    if algorithm == "item to item":
        with open('text_files/item_to_item/user_' + str(user_input) + '_' + metric +'.txt', 'r') as file:
            for line in file:
                key,value = map(float, line.strip().split())
                rx[key] = value

    if algorithm == "tag based":
        with open('text_files/tag_based/movie_' + str(user_input) + '_' + metric +'.txt', 'r') as file:
            for line in file:
                key,value = map(float, line.strip().split())
                rx[key] = value

    if algorithm == "content based":
        with open('text_files/content_based/movie_' + str(user_input) + '_' + metric +'.txt', 'r') as file:
            for line in file:
                key,value = map(float, line.strip().split())
                rx[key] = value
    
    if algorithm == "hybrid":
        inputs = user_input.split()
        with open('text_files/hybrid/score_' + str(inputs[0]) + '_' + str(inputs[1]) + '_' + metric +'.txt', 'r') as file:
            for line in file:
                key,value = map(float, line.strip().split())
                rx[key] = value

    sorted_rx = dict(sorted(rx.items(), key=lambda item: item[1], reverse=True))
    first_n_keys = list(sorted_rx.keys())[:100] # take 100 most recommended movies for our user
    recommended_movies = movies_df[movies_df['movieId'].isin(first_n_keys)]
    recommended_movies_html = recommended_movies.to_html()

    # Pass the recommended movies to the template so I can get the table
    return render_template('recommendation.html', recommended_movies=recommended_movies_html)

if __name__ == '__main__':
    app.run(debug=True)