import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


# Classes to operate our searches
# import DataLoader
# import MosaicRX.backend.Embeddings as Embeddings


# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'init.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)
    df = pd.DataFrame(data)

app = Flask(__name__)
CORS(app)

vectorizer = TfidfVectorizer(stop_words='english')
tf_idf_mat = vectorizer.fit_transform(df['uses'].fillna(""))

def recommend_medicines(query, top_n=10):
    """
    Given a query (symptoms or feelings), compute cosine similarities 
    between the query and the uses column of the medicine data, and return the top_n matches.
    """
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tf_idf_mat).flatten()
    
    top_indices = sims.argsort()[::-1][:top_n]
    
    recommendations = df.iloc[top_indices].copy()
    recommendations['similarity'] = sims[top_indices]
    
    return recommendations

def aggregate_reviews(review_list):
    """
    Compiles the comments from a list of reviews together into one string
    """
    s = ""
    if type(review_list) == list:
        for r in review_list:
            s += " " + r["comment"]
        return s
    return ""

def svd_transformation(recommendations, num_components):
    """
    Applies SVD to sort the given recommendations based on user reviews.
    """
    recommendations['reviews_text'] = recommendations[['user_reviews']].apply(aggregate_reviews)
   
    reviews_texts = recommendations['reviews_text'].tolist()
    reviews_vectorizer = TfidfVectorizer(stop_words='english')
    reviews_tf_idf_mat = reviews_vectorizer.fit_transform(reviews_texts)
   
    svd_reviews = TruncatedSVD(n_components=num_components, random_state=42)
    sentiment_scores = svd_reviews.fit_transform(reviews_tf_idf_mat).flatten()
    recommendations['sentiment_score'] = sentiment_scores
   
    recommendations_sorted = recommendations.sort_values(by='sentiment_score', ascending=False)
    recommendations_sorted = recommendations_sorted.drop(columns=['reviews_text'])
   
    return recommendations_sorted

@app.route("/")
def home():
    return render_template('base.html', title="Medicine Recommender")

@app.route("/recommend", methods=["GET"])
def recommend_search():
    query = request.args.get("query", "")
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    recommendations = recommend_medicines(query, top_n=10)
    return recommendations.to_json(orient="records")

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)