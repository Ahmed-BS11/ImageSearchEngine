import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from elasticsearch import Elasticsearch
from tensorflow import keras
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from skimage.feature import local_binary_pattern
import numpy as np
import cv2
import os
import pandas as pd

def is_valid_image_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return True
    except Exception as e:
        return False

def extract_inceptionv3_features(img):
    img = cv2.resize(img, (299, 299))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    features = model.predict(img)
    return features

def extract_lbp_features(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    radius = 1
    n_points = 8 * radius
    lbp_image = local_binary_pattern(img_gray, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def combine_features(inceptionv3_features, lbp_features):
    return np.concatenate((inceptionv3_features, lbp_features), axis=None)

def compute_euclidean_distance(query_vector, dataset_vectors):
    distances = np.linalg.norm(dataset_vectors - query_vector, axis=1)
    return distances

base_model = InceptionV3(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

es = Elasticsearch("http://localhost:9200")

st.title("Image Search")
index_name = "images_data"

# Initialize lists to store dataset features and paths
dataset_features = []
images_path = []

# Set the batch size for each request
batch_size = 1000

# Define the scroll timeout
scroll_timeout = "1m"  # 1 minute, adjust as needed

# Create an initial search request with scrolling
res = es.search(index=index_name, size=batch_size, scroll=scroll_timeout)

# Extract the initial batch of hits
hits = res['hits']['hits']

while hits:
    for hit in hits:
        features_dense1 = hit['_source']['features_dense1']
        features_dense2 = hit['_source']['features_dense2']
        features_dense3 = hit['_source']['features_dense3']

        # Combine all three feature sets into a single vector
        combined_features = np.concatenate((features_dense1, features_dense2, features_dense3), axis=None)

        # Store the combined features and the image path
        dataset_features.append(combined_features)
        images_path.append(hit['_source']['path'])

    # Perform a scroll request to get the next batch of hits
    res = es.scroll(scroll_id=res['_scroll_id'], scroll=scroll_timeout)
    hits = res['hits']['hits']

# Don't forget to clear the scroll context when done
es.clear_scroll(scroll_id=res['_scroll_id'])


option = st.radio("Select Input Option", ("Upload Image", "Enter Image URL"))

if option == "Upload Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", 'webp'])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        image = np.array(image)
        new_inceptionv3_features = extract_inceptionv3_features(image)
        new_lbp_features = extract_lbp_features(image)
        new_combined_features = combine_features(new_inceptionv3_features, new_lbp_features)

        # Center the "Search" button
        st.markdown("<div style='text-align: center;'><button class='big-button' onclick='search()'>Search</button></div>", unsafe_allow_html=True)

        if st.button("Search", key="search_button", help="Click to perform the search"):
            similarity_scores = compute_euclidean_distance(new_combined_features, dataset_features)
            sorted_indices = np.argsort(similarity_scores)
            top_N = 10
            most_similar_image_paths = [images_path[i] for i in sorted_indices[:top_N]]
            most_similar_scores = [similarity_scores[i] for i in sorted_indices[:top_N]]

            for img_path, score in zip(most_similar_image_paths, most_similar_scores):
                st.subheader(f"Image Path: {img_path}, Similarity Score: {score}")

                if os.path.isfile(img_path):  # Check if the image file exists locally
                    similar_image = Image.open(img_path)
                    st.subheader("Most Similar Image:")
                    st.image(similar_image, use_column_width=True)
                else:
                    st.write("Image file not found at the specified path")