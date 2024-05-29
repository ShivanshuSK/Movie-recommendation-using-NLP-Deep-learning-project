import matplotlib.pyplot as plt
import pandas as pd
import tensorflow_hub as hub
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(model_url)
print('Model Loaded')


def embed(texts):
    return model(texts)


embed(['This movie was great!'])

df = pd.read_csv("C:\Users\Asus\PycharmProjects\MOVIE RECOMMENDATION USING nlp\Top_10000_Movies.csv", engine="python")
df.head()

df = df[["original_title", "overview"]]
df.head()
df = df.dropna()
df = df.reset_index()
df = df[:5500]

titles = list(df['overview'])
var = titles[:5]
embeddings = embed(titles)
print('The embedding shape is:', embeddings.shape)
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(embeddings)
plt.figure(figsize=(11, 6))
plt.title('Embedding space')
plt.scatter(emb_2d[:, 0], emb_2d[:, 1])
plt.show()
nn = NearestNeighbors(n_neighbors=10)
nn.fit(embeddings)


def recommend(text):
    emb = embed([text])
    neighbors = nn.kneighbors(emb, return_distance=False)[0]
    return df['original_title'].iloc[neighbors].tolist()


print('Recommended Movies:')
recommend("sci-fi")
