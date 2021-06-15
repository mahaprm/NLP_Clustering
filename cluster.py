import pickle

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

# nltk.download('stopwords')
# nltk.download('wordnet')

# Cleaning the text

import string

df = pd.read_csv("dataset/test.csv")

original_text = df['original_text']


def text_process(text):
    stemmer = WordNetLemmatizer()
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join([i for i in nopunc if not i.isdigit()])
    nopunc = [word.lower() for word in nopunc.split() if word not in stopwords.words('english')]
    return [stemmer.lemmatize(word) for word in nopunc]


# Vectorisation : -

from sklearn.feature_extraction.text import TfidfVectorizer

vector = TfidfVectorizer(analyzer=text_process, ngram_range=(1, 3))

X_transformed = vector.fit_transform(original_text)

# Clustering the training sentences with K-means technique

from sklearn.cluster import KMeans

 modelkmeans = KMeans(n_clusters=4, init='k-means++', n_init=100)
 modelkmeans.fit(X_transformed)

 labels = modelkmeans.labels_
 centroids = modelkmeans.cluster_centers_

 print(labels)
 print(centroids)

# # save the model to disk
filename = 'finalized_model.sav'
# pickle.dump(modelkmeans, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))

predicted = loaded_model.predict(vector.transform(df['original_text']))

df.loc[:, 'cluster'] = pd.Series(predicted)

# df = df['cluster'] = pd.Series(predicted)

print(type(predicted))

predicted_df = df[['id', 'cluster']]

predicted_df['cluster'] = predicted_df['cluster'].apply(lambda x: 'Type-' + np.str_(x))

print(predicted_df.head())

predicted_df.to_csv('submission.csv')
