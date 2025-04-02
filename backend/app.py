import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from embeddings import *
from data_loader import *
from recommender import *
from clustering import *


# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'inittest.json')

app = Flask(__name__)
CORS(app)

data_loader = DataLoader(json_file=json_file_path)
documents = data_loader.dataset()
embeddings = Embeddings(documents=documents)
clusterer = Clustering(document_embeddings=embeddings.get_document_embeddings())
recommendation_system = Recommender(embeddings=embeddings, data_loader=data_loader, clusterer=clusterer)

@app.route("/")
def home():
    return render_template('base.html', title="Medicine Recommender")

@app.route("/recommend", methods=["GET"])
def recommend_search():
    query = request.args.get("query", "")
    if not query:
        return jsonify({"error": "No query provided"}), 400

    recommendations = recommendation_system.generate_recommendations(query, k=10)
    return jsonify(recommendations)
   
if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)