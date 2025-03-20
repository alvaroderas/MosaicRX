import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
tf_idf_mat = vectorizer.fit_transform(df['Uses'].fillna(""))

def recommend_medicines(query, top_n=5):
    """
    Given a query string (e.g., symptoms), compute cosine similarities 
    between the query and the "Uses" column of the medicine data, and return the top_n matches.
    """
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tf_idf_mat).flatten()
    
    top_indices = sims.argsort()[::-1][:top_n]
    
    recommendations = df.iloc[top_indices].copy()
    recommendations['similarity'] = sims[top_indices]
    
    return recommendations

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