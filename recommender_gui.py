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

def create_scrollable_window(title, content_text):
    global new_window
    if new_window:
        new_window.destroy()

    new_window = Toplevel(master)
    new_window.title(title)

    # Create a Canvas widget that covers the entire window
    canvas = Canvas(new_window)
    canvas.pack(side="left", fill="both", expand=True)

    # Create vertical scrollbar
    v_scrollbar = Scrollbar(new_window, orient="vertical", command=canvas.yview)
    v_scrollbar.pack(side="right", fill="y")

    # Configure the canvas to use the vertical scrollbar
    canvas.config(yscrollcommand=v_scrollbar.set)

    # Add content to the canvas
    content_frame = Frame(canvas)
    canvas.create_window((0, 0), window=content_frame, anchor="nw")

    # Add a Label to the content frame with the provided text

    content_label = Label(content_frame, text=content_text, font=custom_font)
    content_label.pack()

    # Update the canvas scroll region when the label size changes
    content_label.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))



def fetch_recommended_movies():
    user_input = entry.get()

    rx = {}
    movies_df = pd.read_csv(arguments[1] + '/movies.csv') # load this to show recommended movies as output
    
    if algorithm_variable.get() == "user to user":
         with open('text_files/user_to_user/user_' + str(user_input) + '_' + metric_variable.get() +'.txt', 'r') as file:
            for line in file:
                key,value = map(float, line.strip().split())
                rx[key] = value

    if algorithm_variable.get() == "item to item":
        with open('text_files/item_to_item/user_' + str(user_input) + '_' + metric_variable.get() +'.txt', 'r') as file:
            for line in file:
                key,value = map(float, line.strip().split())
                rx[key] = value

    if algorithm_variable.get() == "tag based":
        with open('text_files/tag_based/movie_' + str(user_input) + '_' + metric_variable.get() +'.txt', 'r') as file:
            for line in file:
                key,value = map(float, line.strip().split())
                rx[key] = value

    if algorithm_variable.get() == "content based":
        with open('text_files/content_based/movie_' + str(user_input) + '_' + metric_variable.get() +'.txt', 'r') as file:
            for line in file:
                key,value = map(float, line.strip().split())
                rx[key] = value
    
    if algorithm_variable.get() == "hybrid":
        inputs = user_input.split()
        print(inputs)
        with open('text_files/hybrid/score_' + str(inputs[0]) + '_' + str(inputs[1]) + '_' + metric_variable.get() +'.txt', 'r') as file:
            for line in file:
                key,value = map(float, line.strip().split())
                rx[key] = value

    sorted_rx = dict(sorted(rx.items(), key=lambda item: item[1], reverse=True))
    first_n_keys = list(sorted_rx.keys())[:100]
    recommended_movies = movies_df[movies_df['movieId'].isin(first_n_keys)]
    movies_list = "\n".join(recommended_movies['title'].values)

    # Display recommended movies in a new window with a scrollbar
    create_scrollable_window("Recommended movies", movies_list)


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

button = Button(master, text="Search", command=fetch_recommended_movies,font=custom_font)
button.pack()

# Initialize global variable to destroy window each time the Search button is pushed
new_window = None

mainloop()