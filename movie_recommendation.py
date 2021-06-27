import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# functions 
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]


#Step 1: Read File
df=pd.read_csv("movie_dataset.csv") 


#Step 2: Select Features
features = ['keywords','cast','genres','director']
for feature in features:
	df[feature] = df[feature].fillna('')


#Step 3:combining all features
def combine_features(row): 
		return row["keywords"] +" "+row["cast"] +" "+row["genres"] +" "+row["director"] 
df["combined_features"] = df.apply(combine_features,axis=1)


#Step 4: extracting features from dataset
cv=CountVectorizer()
count_matrix=cv.fit_transform(df["combined_features"])

#Step 5: using cosine similarity
cosine_sim = cosine_similarity(count_matrix)


# Step 6: Get index of this movie from its title
movie_user_likes = "The Wood"
movie_index = get_index_from_title(movie_user_likes)


#Step 7: generate the similar movie matrix and sorted in descending order(similarity score)
similar_movies = list(enumerate(cosine_sim[movie_index]))
sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)


#Step 8: printing the similar movies
i=0
print("Top 10 similar movies to "+movie_user_likes+" are:\n")
for movie in sorted_similar_movies:
	print(get_title_from_index(movie[0]))
	i=i+1
	if i>10:
		break

print("\n")	
