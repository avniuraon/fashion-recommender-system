#  Fashion Recommender System

A deep learning-based fashion recommendation system that suggests visually similar clothing items using image embeddings.

##  Overview

This project uses **ResNet50** (pretrained on ImageNet) to extract features from clothing images and recommend similar items based on visual similarity.

Instead of relying on metadata, the system understands **style, texture, and patterns directly from images**.

---

##  How It Works

1. **Feature Extraction**
   - Uses ResNet50 to convert images into high-dimensional feature vectors (embeddings)

2. **Similarity Search**
   - Applies K-Nearest Neighbors (KNN) to find visually similar items

3. **Recommendation**
   - Given an input image, returns the top 5 most similar fashion items

---

##  Tech Stack

- Python  
- TensorFlow / Keras  
- ResNet50 (ImageNet pretrained)  
- Scikit-learn (KNN)  
- NumPy  
- HTML, CSS (Frontend)

---

##  Features

- Image-based recommendations (no tags needed)
- Efficient similarity search using embeddings
- Clean and simple UI for interaction
- Scalable pipeline for adding new items

---

##  Project Structure


---

##  Future Improvements

- Add user personalization
- Deploy as a web app
- Use approximate nearest neighbors (FAISS) for faster search
- Improve UI/UX

---

##  Key Learning

- Learned how to use pretrained CNNs (ResNet50) for feature extraction
- Understood image embeddings and similarity search
- Built an end-to-end ML application pipeline

---

##  Demo

_Add screenshots or demo here_

---

##  Contributing

Feel free to fork the repo and improve the project!

---

##  If you like this project, give it a star!
