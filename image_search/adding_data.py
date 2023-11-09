from elasticsearch import Elasticsearch, exceptions
from elasticsearch import helpers
from elasticsearch.helpers import bulk
import pandas as pd

# Connect to Elasticsearch
es = Elasticsearch(hosts=['localhost'], port=9200)

# Define the index settings and mappings
try:
    index_name = 'images_data'
    settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "mappings": {
            "dynamic": False,
            "properties": {
                "path": {"type": "text"},
                "vector": {
                    "type": "elastiknn_dense_float_vector",
                    "elastiknn": {
                        "dims": 2058,
                        "similarity": "L2",  # Use "l2" for Euclidean distance
                        "model": "lsh",
                        "L": 99,
                        "k": 1,
                        "w": 3
                    }
                },
            }
        }
    }

    # Create the index
    if es.indices.exists(index=index_name):
        # Delete the index
        es.indices.delete(index=index_name)
        print(f"Index '{index_name}' has been deleted.")
    else:
        print(f"Index '{index_name}' does not exist.")

    es.indices.create(index=index_name, body=settings)

    # Read data from CSV file
    data = pd.read_csv(r"C:\Users\ahmed\Desktop\search_engine\Final_Data.csv")

    # Prepare the data for bulk indexing
    bulk_data = []
    for _, row in data.iterrows():
        doc = {
            "_index": index_name,
            "_source": {
                "path": row.iloc[0],
                # Assuming that the "vector" field matches the mapping
                "vector": row.iloc[1:].values.tolist()
            }
        }
        bulk_data.append(doc)

    # Perform bulk indexing
    helpers.bulk(es, bulk_data)

except exceptions.RequestError as e:
    # The exception contains information about the mapping issue
    error_info = e.info
    print("Mapping-related error info:")
    print(error_info)
