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

# Load the InceptionV3 model
model = load_inceptionv3_model()

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

# Define a slider for the user to choose the number of pictures to show

@st.cache_data(experimental_allow_widgets=True)  # 👈 Set the parameter
def get_num_results():
    num_results = st.slider("Number of Pictures to Show", min_value=1, max_value=30, value=10)
    return num_results

num_results=get_num_results()
option = st.radio("Select Input Option", ("Upload Image", "Enter Image URL","Search By Tags"))

if option == "Upload Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", 'webp'])

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        image = np.array(image)
        new_inceptionv3_features = extract_inceptionv3_features(image)
        new_lbp_features = extract_lbp_features(image)
        new_combined_features = combine_features(new_inceptionv3_features, new_lbp_features)


        # Flag to control the search loop
        cancel_search = False

        if st.button("Search", key="search_button", help="Click to perform the search"):
            # Clear any previous search results
            st.spinner()
            with st.spinner(text="Searching..."):
                similarity_scores = compute_euclidean_distance(new_combined_features, dataset_features)
                sorted_indices = np.argsort(similarity_scores)
                top_N = num_results
                most_similar_image_paths = [images_path[i] for i in sorted_indices[:top_N]]
                most_similar_scores = [similarity_scores[i] for i in sorted_indices[:top_N]]

                for img_path, score in zip(most_similar_image_paths, most_similar_scores):
                    if cancel_search:
                        st.write("Search canceled.")
                        break  # Exit the loop if canceled
                    st.subheader(f"Image Path: {img_path}, Similarity Score: {score}")

                    if os.path.isfile(img_path):  # Check if the image file exists locally
                        similar_image = Image.open(img_path)
                        st.subheader("Most Similar Image:")
                        st.image(similar_image, use_column_width=True)
                    else:
                        st.write("Image file not found at the specified path")

                if cancel_search:
                    st.write("Search canceled.")
                else:
                    st.write("Search complete.")

        # Add a "Cancel" button
        if st.button("Cancel Search", key="cancel_button"):
            cancel_search = True

if option == "Enter Image URL":
    # Text input for entering an image URL
    image_url = st.text_input("Enter Image URL")
    if image_url:
        try:
            # Send an HTTP GET request to the image URL
            response = requests.get(image_url)
            response.raise_for_status()  # Raise an exception for invalid URLs
            
            # Check if the response content is an image
            image = Image.open(BytesIO(response.content))
            image = Image.open(BytesIO(response.content)).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            image = np.array(image)
            new_inceptionv3_features = extract_inceptionv3_features(image)
            new_lbp_features = extract_lbp_features(image)
            new_combined_features = combine_features(new_inceptionv3_features, new_lbp_features)

            if st.button("Search", key="search_button", help="Click to perform the search"):
                # Clear any previous search results
                st.spinner()
                with st.spinner(text="Searching..."):
                    similarity_scores = compute_euclidean_distance(new_combined_features, dataset_features)
                    sorted_indices = np.argsort(similarity_scores)
                    
                    # Limit the number of search results based on the user's selection
                    max_search_results = min(num_results, len(dataset_features))
                    most_similar_image_paths = [images_path[i] for i in sorted_indices[:max_search_results]]
                    most_similar_scores = [similarity_scores[i] for i in sorted_indices[:max_search_results]]

                    for img_path, score in zip(most_similar_image_paths, most_similar_scores):
                        st.subheader(f"Image Path: {img_path}, Similarity Score: {score}")

                        if os.path.isfile(img_path):  # Check if the image file exists locally
                            similar_image = Image.open(img_path)
                            st.subheader("Most Similar Image:")
                            st.image(similar_image, use_column_width=True)
                        else:
                            st.write("Image file not found at the specified path")

                    st.write("Search complete.")
        except Exception as e:
            st.write("Error: Invalid Image URL or unable to retrieve image.")

if option == "Search By Tags":
    tags_input = st.text_input("Enter Tags (comma-separated)", "")

# Search button
    if st.button("Search") or tags_input:
        # Split the input tags by commas and trim spaces
        tags = [tag.strip() for tag in tags_input.split(",")]

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
