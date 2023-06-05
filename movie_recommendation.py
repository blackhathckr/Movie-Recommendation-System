import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

org_movies = pd.read_csv('movies_dataset.csv',error_bad_lines=False,engine='python',encoding='utf-8')
movies = org_movies[[ 'genres', 'keywords','cast', 'title', 'director']]

movies.fillna('',inplace=True)

movies['combined_features'] = movies['genres'] +' '+ movies['keywords'] +' '+ movies['cast'] +' '+ movies['title'] +' '+ movies['director']
#movies.iloc[0]['combined_features']

cv = CountVectorizer()
count_matrix = cv.fit_transform(movies['combined_features'])

cs = cosine_similarity(count_matrix)

def get_movie_name_from_index(index):
    return org_movies[org_movies['index']==index]['title'].values[0]

def get_index_from_movie_name(name):
    return org_movies[org_movies['title']==name]['index'].values[0]

def recommend(movie):
	movies_list=list(movies['title'])

	if movie not in movies_list:
		st.title("Movie not Found !")
	else:
		test_movie_index = get_index_from_movie_name(movie)
		movie_corrs = cs[test_movie_index]
		movie_corrs = enumerate(movie_corrs)
		sorted_similar_movies = sorted(movie_corrs,key=lambda x:x[1],reverse=True)
		for i in range(10):
			st.title(get_movie_name_from_index(sorted_similar_movies[i][0]))
                

def movie_recommender_webapp():
    st.title("Movie Recommendation Engine Web App")
    movie=st.text_input("Enter the Movie Name for Recommendations")
    if st.button("Recommend"):
	    recommend(movie)
         
movie_recommender_webapp()
           



