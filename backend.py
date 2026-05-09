import pickle
import numpy as np
from numpy.linalg import norm
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import io
import tensorflow as tf

app = FastAPI()

# ---- CORS (VERY IMPORTANT or frontend will cry) ----

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- load data ----

feature_list = pickle.load(open("embeddings.pkl", "rb"))
filenames = pickle.load(open("filenames.pkl", "rb"))

# ---- load model ----

base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = tf.keras.Sequential([
base_model,
GlobalMaxPooling2D()
])

# ---- nearest neighbors ----

neighbors = NearestNeighbors(n_neighbors=6, metric="euclidean")
neighbors.fit(feature_list)

# ---- serve images ----

app.mount("/images", StaticFiles(directory="fashion-dataset/images"), name="images")

# ---- helper function ----

# ---- helper function ----
def extract_features(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img)
    expanded = np.expand_dims(img_array, axis=0)
    preprocessed = preprocess_input(expanded)

    result = model.predict(preprocessed).flatten()
    normalized = result / norm(result)

    return normalized


# ---- API endpoint ----
@app.post("/recommend")
async def recommend(file: UploadFile = File(...)):
    img_bytes = await file.read()

    features = extract_features(img_bytes)

    distances, indices = neighbors.kneighbors([features])

    recommended_files = []
    for i in indices[0][1:6]:
        path = filenames[i]

        # convert local path -> URL path
        filename = path.split("/")[-1].split("\\")[-1]
        recommended_files.append(f"/images/{filename}")

    return {"recommendations": recommended_files}