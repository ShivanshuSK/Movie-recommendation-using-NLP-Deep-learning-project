import pickle
import numpy as np
import pandas as pd
import tensorflow_hub as hub
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from flask import Flask,render_template,request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    ans=[]
    if request.method=='POST':
        with open('embeddings.pkl', 'rb') as file:
            embeddings = pickle.load(file)

        with open('nearest_neighbors_model.pkl', 'rb') as file:
            nn = pickle.load(file)

        def recommend(text):
            emb = embed([text])
            neighbors = nn.kneighbors(emb, return_distance=False)[0]
            return df['original_title'].iloc[neighbors].tolist()
        var1=request.form['userInput']
        ans=recommend(var1)

    return render_template('index.html', ans=ans)


if __name__ == '__main__':
    app.run(debug=True)