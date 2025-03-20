import json
import os
from flask import Flask, render_template, request, jsonify
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
from flask_cors import CORS
from DataLoader import DataLoader
from Embeddings import Embeddings
from Recommender import Recommender


# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'init.json')

# Assuming your JSON data is stored in a file named 'init.json'
# with open(json_file_path, 'r') as file:
#     data = json.load(file)
#     df = pd.DataFrame(data)

app = Flask(__name__)
CORS(app)



dataloader = DataLoader(json_file=json_file_path)
embeddings = Embeddings(dataloader)
recommender = Recommender(dataloader, embeddings)



@app.route("/")
def home():
    return render_template('base.html', title="Medicine Recommender")


@app.route("/recommend", methods=["GET"])
def recommend_search():
    query = request.args.get("query", "")
    if not query:
        return jsonify({"error": "No query provided"}), 400
    recommendations = recommender.search(query)
    return jsonify(recommendations)
   
if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)