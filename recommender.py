import pandas as pd
import numpy as np


ratings_df = pd.read_csv('datasets/ratings.csv')
tags_df = pd.read_csv('datasets/tags.csv')

print('ratings_df size: ', ratings_df.size)
print('tags_df size: ', tags_df.size)

def jaccard(list1, list2):
    """Define Jaccard Similarity function for two sets"""
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

def dice(x,y):
    return (2*len(x.intersection(y))) / (len(x) + len(y)) 

