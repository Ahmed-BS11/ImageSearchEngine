import streamlit as st
from elasticsearch import Elasticsearch
from PIL import Image
import requests
from io import BytesIO

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
# Create an Elasticsearch client
es = Elasticsearch([{'host': '127.0.0.1', 'port': 9200}])

# Streamlit app title
st.title("Elasticsearch Search Interface")

# Text input for entering tags
tags_input = st.text_input("Enter Tags (comma-separated)", "")

# Search button
if st.button("Search"):
    # Split the input tags by commas and trim spaces
    tags = [tag.strip() for tag in tags_input.split(",")]

    # Define the search query based on the entered tags
    search_body = {
        "size": 100,  # Adjust the size as needed
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
