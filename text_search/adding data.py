import csv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

# Create an Elasticsearch client
es = Elasticsearch([{'host': '127.0.0.1', 'port': 9200}])

# Specify the index name
index_name = 'flickrphotos'

# Define the CSV file path
csv_file_path = r"C:\Users\ahmed\Desktop\atelier1\photo_metadata.csv"

# Create a generator to yield documents for bulk indexing
def generate_documents():
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            yield {
                '_index': index_name,
                '_source': row,
            }

# Use the bulk API to perform bulk indexing
success, failed = bulk(es, generate_documents())

# Refresh the index to make the documents available for searching
es.indices.refresh(index=index_name)

# Print the number of successfully indexed documents and any failures
print(f"Successfully indexed: {success}")
print(f"Failures: {failed}")
