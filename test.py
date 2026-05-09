import pickle
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import cv2

feature_list = pickle.load(open('embeddings.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

img = image.load_img("fashion-dataset/sample/cool.jpg", target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img)
result = model.predict(preprocessed_img).flatten()
normalised_result = result / norm(result)

neighbors = NearestNeighbors(n_neighbors=6, metric='euclidean')
neighbors.fit(feature_list)

distances, indices = neighbors.kneighbors([normalised_result])

print(indices)

for file in indices[0][1:6]:
    temp_img = cv2.imread(filenames[file])
    cv2.imshow("Result", cv2.resize(temp_img, (512, 512)))
    cv2.waitKey(0)