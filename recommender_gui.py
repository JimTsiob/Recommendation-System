from tkinter import *
from tkinter import font
import pandas as pd 
import sys


METRICS = [
"jaccard",
"dice",
"cosine",
"pearson"
] #similarity metrics

ALGORITHMS = [
    "user to user",
    "item to item",
    "tag based",
    "content based",
    "hybrid"
] # algorithms

master = Tk() # master = the window widget
master.geometry('1366x800') # dimensions of window
master.title('Recommender GUI') # title of window
custom_font = font.Font(family="Helvetica", size=12)

arguments = sys.argv[1:]
if len(arguments) < 2:
    print("ERROR: please provide all arguments.")
    print('example: python recommender_gui.py -d datasets')
    master.destroy()

metric_variable = StringVar(master)
metric_variable.set(METRICS[0]) # default value for metrics dropdown

dropdown_metrics = OptionMenu(master, metric_variable, *METRICS)
dropdown_metrics.config(font=custom_font)
dropdown_metrics.pack() # dropdown with metrics, pack function adds dropdown to widget.

algorithm_variable = StringVar(master)
algorithm_variable.set(ALGORITHMS[0])

dropdown_algorithms = OptionMenu(master,algorithm_variable, *ALGORITHMS)
dropdown_algorithms.config(font=custom_font)
dropdown_algorithms.pack() # dropdown with algorithms

entry = Entry(master, width=30,font=custom_font)

# Pack the Entry and Button widgets into the main window
entry.pack(pady=10)  # pady adds some vertical padding



def fetch_recommended_movies():
    
    user_input = entry.get()
    # custom_font = font.Font(family="Helvetica", size=12)
    movies_df = pd.read_csv(arguments[1] + '/movies.csv') # load this to show recommended movies as output

    rx = {}
    with open('text_files/user_to_user/user_' + str(user_input) + '_' + metric_variable.get() +'.txt', 'r') as file:
            for line in file:
                key,value = map(float, line.strip().split())
                rx[key] = value
        
    sorted_rx = dict(sorted(rx.items(), key=lambda item: item[1], reverse=True)) # sort recommendation scores in descending order
    first_n_keys = list(sorted_rx.keys())[:100] # get top 100 keys
    recommended_movies = movies_df[movies_df['movieId'].isin(first_n_keys)]

    # canvas = Canvas(master)
    # canvas.pack(expand=True, fill="both") # this is added for scrollbar container

    top100_label.config(text="\nHere are your top " + str(100) + " recommendations: \n",font=custom_font)
    movies_list = "\n".join(recommended_movies['title'].values)

    # scrollbar = Scrollbar(master, command=canvas.yview)
    # scrollbar.pack(side="right", fill="y")

    result_label.config(text=movies_list,font=custom_font) # print top 100 movies in the window

    # canvas.config(yscrollcommand=scrollbar.set)
    # canvas.create_window((0, 0), window=result_label, anchor="nw")

    # result_label.update_idletasks()
    # canvas.config(scrollregion=canvas.bbox("all"))
    # print(recommended_movies['title'].values)
    # result_label.config(text="value is: " + metric_variable.get() + " - " + algorithm_variable.get() + ' -> user: ' + user_input,font=custom_font)
    # print ("value is: " + metric_variable.get() + " - " + algorithm_variable.get() + ' -> user: ' + user_input)

button = Button(master, text="Search", command=fetch_recommended_movies,font=custom_font)
button.pack()

top100_label = Label(master, text="",font=custom_font)
top100_label.pack(pady=10)

result_label = Label(master, text="",font=custom_font)
result_label.pack(pady=10)

mainloop()