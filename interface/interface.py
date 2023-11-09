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
import requests
from io import BytesIO
from autocorrect import Speller
import json  # Import the json module

spell = Speller(lang='en')

def is_valid_image_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return True
    except Exception as e:
        return False
    
@st.cache_data
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

@st.cache_data
def compute_euclidean_distance(query_vector, dataset_vectors):
    distances = np.linalg.norm(dataset_vectors - query_vector, axis=1)
    return distances

@st.cache_resource
def load_inceptionv3_model():
    base_model = InceptionV3(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    return model

@st.cache_data(experimental_allow_widgets=True)  # 👈 Set the parameter
def get_num_results():
    num_results = st.sidebar.slider("Number of Pictures to Show", min_value=1, max_value=30, value=10)
    return num_results

num_results=get_num_results()

def search_by_image_query(feature_vector=None, size=num_results):
    if feature_vector is None:
        raise ValueError("Please enter an Image ID or a Feature Vector")

    # Convert the NumPy array to a Python list
    feature_vector = feature_vector.tolist()

    query = {
        "query": {
            "elastiknn_nearest_neighbors": {
                "vec": feature_vector,
                "field": "vector",
                "similarity": "l2",
                "model": "lsh",
                "candidates": 10
            }
        }
    }

    res = es.search(index=index_name, body=query, size=size)

    search_results = []
    for hit in res["hits"]["hits"]:
        path = hit["_source"]["path"]
        image = Image.open(path)
        search_results.append(image)

    if search_results:
        st.write("First 3 by 3 similar images:")
        columns = st.columns(3)

        for i, result in enumerate(search_results[:size]):
            with columns[i % 3]:
                st.image(result, caption=f"Result {i + 1}", use_column_width=True)
    else:
        st.write("No similar images found.")
st.image("logo.png", use_column_width=False, width=200)

# Add a brief introduction to your project
st.title("Image and Text Search Engine")

st.write("This project demonstrates a comprehensive search engine that allows users to search for images based on content, tags, and more.")
# Load the InceptionV3 model
model = load_inceptionv3_model()

es = Elasticsearch("http://localhost:9200")

index_name = "images_data"

# Define a slider for the user to choose the number of pictures to show
custom_css = """
    <style>
        .custom-button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            font-size: 18px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }
    </style>
"""

# Define a radio button for selecting the input option
option = st.sidebar.radio("Select Input Option", ("Upload Image", "Enter Image URL", "Search By Tags"))
if option == "Upload Image":
    uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png", 'webp'])

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        image = np.array(image)
        new_inceptionv3_features = extract_inceptionv3_features(image)
        new_lbp_features = extract_lbp_features(image)
        new_combined_features = combine_features(new_inceptionv3_features, new_lbp_features)
        if st.sidebar.button("Search", key="search_button", help="Click to perform the search"):
            # Clear any previous search results
            st.spinner()
            with st.spinner(text="Searching..."):
                # Use the search_by_image_query function to perform the search
                search_by_image_query(new_combined_features, size=num_results)


if option == "Enter Image URL":
    # Text input for entering an image URL
    image_url = st.sidebar.text_input("Enter Image URL")
    if image_url:
        try:
            # Send an HTTP GET request to the image URL
            response = requests.get(image_url)
            response.raise_for_status()  # Raise an exception for invalid URLs
            
            # Check if the response content is an image
            image = Image.open(BytesIO(response.content)).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            image = np.array(image)
            new_inceptionv3_features = extract_inceptionv3_features(image)
            new_lbp_features = extract_lbp_features(image)
            new_combined_features = combine_features(new_inceptionv3_features, new_lbp_features)

            if st.sidebar.button("Search", key="search_button", help="Click to perform the search"):
                # Clear any previous search results
                st.spinner()
                with st.spinner(text="Searching..."):
                    # Use the search_by_image_query function to perform the search
                    search_by_image_query(new_combined_features, size=num_results)
        except Exception as e:
            st.write("Error: Invalid Image URL or unable to retrieve")


if option == "Search By Tags":
    tags_input = st.sidebar.text_input("Enter Tags (comma-separated)", "")

# Search button
    if st.button("Search") or tags_input:
        # Split the input tags by commas and trim spaces
        tags = [tag.strip() for tag in tags_input.split(",")]
        tags = [spell(tag) for tag in tags]
        # Define the search query based on the entered tags
        search_body = {
            "size": num_results,  # Adjust the size as needed
            "query": {
                "terms": {
                    "tags": tags
                }
            }
        }

        # Specify the index to search (in this case, 'flickrphotos')
        index_name = 'flickrphotos'

        # Perform the search
        try:
            response = es.search(index=index_name, body=search_body)
            hits = response['hits']['hits']

            # Display search results with columns
            st.subheader("Search Results:")

            # Specify the number of images per column
            images_per_column = 3

            for i in range(0, len(hits), images_per_column):
                column = st.columns(images_per_column)
                for j in range(i, min(i + images_per_column, len(hits))):
                    hit = hits[j]
                    source = hit['_source']
                    with column[j % images_per_column]:
                        st.write(f"Title: {source.get('title', 'N/A')}")
                        #st.write(f"Tags: {source.get('tags', 'N/A')}")
                        farm = source.get('flickr_farm', 'N/A')
                        server = source.get('flickr_server', 'N/A')
                        photo_id = source.get('id', 'N/A')
                        secret = source.get('flickr_secret', 'N/A')
                        image_url = f"http://farm{farm}.staticflickr.com/{server}/{photo_id}_{secret}.jpg"
                        if is_valid_image_url(image_url):
                            st.image(image_url, caption=f"Image for {source.get('title', 'N/A')}")
                        else:
                            continue
            st.write(f"Total hits: {len(hits)}")

        except Exception as e:
            st.error(f"Error: {e}")
