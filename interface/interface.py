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

        # Send an HTTP GET request to the image URL

        response = requests.get(url)

        response.raise_for_status()  # Raise an exception for invalid URLs

 

        # Check if the response content is an image

        image = Image.open(BytesIO(response.content))

        return True

    except Exception as e:

        # An exception occurred, indicating an invalid URL or non-image content

        return False

   

# Preprocess input image and extract features

def extract_inceptionv3_features(img):

    img = cv2.resize(img, (299, 299))  # InceptionV3 input size

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

    hist /= (hist.sum() + 1e-7)  # Normalize the histogram

    return hist

 

# Combine InceptionV3 and LBP features

def combine_features(inceptionv3_features, lbp_features):

    return np.concatenate((inceptionv3_features, lbp_features), axis=None)

def compute_euclidean_distance(query_vector, dataset_vectors):

    # Compute the Euclidean distance between the query vector and each dataset vector

    distances = np.linalg.norm(dataset_vectors - query_vector, axis=1)

    return distances

 

# Load pre-trained InceptionV3 model

base_model = InceptionV3(weights='imagenet')

model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)  # Remove the final softmax layer

 

# Create Elasticsearch client

es = Elasticsearch("http://localhost:9200")

 

# Streamlit app title

st.title("Image Search")

# Retrieve indexed images and their features from Elasticsearch

res = es.search(index="images_data")  # Modify "your_index_name" to match your index name
hits = res['hits']['hits']

# Extract the paths and features for each hit
images_path = [hit['_source']['path'] for hit in hits]
dataset_features = [hit['_source']['features_dense1'] for hit in hits]

 

# Option to upload or enter image URL

option = st.radio("Select Input Option", ("Upload Image", "Enter Image URL"))

 

if option == "Upload Image":

    # Upload image

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

 

    if uploaded_image is not None:

        # Display the uploaded image

        image = Image.open(uploaded_image)

        st.image(image, caption="Uploaded Image", use_column_width=True)

 

        image = np.array(image)

        new_inceptionv3_features = extract_inceptionv3_features(image)

        new_lbp_features = extract_lbp_features(image)

 

        # Combine the InceptionV3 and LBP features into a single vector

        new_combined_features = combine_features(new_inceptionv3_features, new_lbp_features)

 

        # Compute similarity scores between the new image and dataset images

        similarity_scores = compute_euclidean_distance(new_combined_features, dataset_features)

 

        # Sort images by similarity scores (higher scores are more similar)

        sorted_indices = np.argsort(similarity_scores)

 

        # Get the top 10 most similar images

        top_N = 1

        most_similar_image_paths = [images_path[i] for i in sorted_indices[:top_N]]

        most_similar_scores = [similarity_scores[i] for i in sorted_indices[:top_N]]

 

        # Display the most similar images with their similarity scores

        for img_path, score in zip(most_similar_image_paths, most_similar_scores):

            print(f"Image Path: {img_path}, Similarity Score: {score}")

            if is_valid_image_url(img_path):

                similar_image = Image.open(requests.get(img_path, stream=True).raw)

                st.subheader("Most Similar Image:")

                st.image(similar_image, use_column_width=True)

        else:

                st.write("Invalid Image URL")