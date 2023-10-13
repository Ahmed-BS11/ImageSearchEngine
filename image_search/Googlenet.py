import os
import cv2
import numpy as np
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from skimage.feature import local_binary_pattern
import pandas as pd

class ImageFeatureExtractor:
    def __init__(self):
        self.base_model = InceptionV3(weights='imagenet')
        self.model = Model(inputs=self.base_model.input, outputs=self.base_model.layers[-2].output)
        
    def extract_inceptionv3_features(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (299, 299))
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        features = self.model.predict(img)
        return features
    
    def extract_lbp_features(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        radius = 1
        n_points = 8 * radius
        lbp_image = local_binary_pattern(img, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        return hist

    def get_image_files(self, root_dir):
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
        image_files = []

        for category in os.listdir(root_dir):
            category_path = os.path.join(root_dir, category)
            if os.path.isdir(category_path):
                for root, _, files in os.walk(category_path):
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in image_extensions):
                            image_files.append(os.path.join(root, file))

        return image_files

    def combine_features(self, inceptionv3_features, lbp_features):
        return np.concatenate((inceptionv3_features, lbp_features), axis=None)

    def compute_euclidean_distance(self, query_vector, dataset_vectors):
        distances = np.linalg.norm(dataset_vectors - query_vector, axis=1)
        return distances

    def extract_and_combine_features(self, image_path):
        inceptionv3_features = self.extract_inceptionv3_features(image_path)
        lbp_features = self.extract_lbp_features(image_path)
        return self.combine_features(inceptionv3_features, lbp_features)

    def build_feature_vectors_csv(self, root_directory, output_file):
        image_files = self.get_image_files(root_directory)
        final_vector_list = []

        for image_path in image_files:
            final_vector = self.extract_and_combine_features(image_path)
            final_vector_list.append(final_vector)

        feature_vector_df = pd.DataFrame(final_vector_list)
        feature_vector_df.to_csv(output_file, index=False)

    def search_similar_images(self, query_image_path, dataset_feature_csv, top_N=10):
        dataset_features = pd.read_csv(dataset_feature_csv)
        new_combined_features = self.extract_and_combine_features(query_image_path)
        similarity_scores = self.compute_euclidean_distance(new_combined_features, dataset_features)

        sorted_indices = np.argsort(similarity_scores)
        most_similar_indices = sorted_indices[:top_N]

        return [image_files[i] for i in most_similar_indices], [similarity_scores[i] for i in most_similar_indices]

if __name__ == "__main__":
    feature_extractor = ImageFeatureExtractor()
    
    # Example: Extract features and build a feature vectors CSV
    #feature_extractor.build_feature_vectors_csv(
    #    r"C:\Users\ahmed\Desktop\Supcom\INDP3_AIM\cbir\bdimage\image_db",
    #    "feature_vectors.csv")
    

    

    # Example: Search for similar images
    similar_image_paths, similarity_scores = feature_extractor.search_similar_images(
        r"C:\Users\ahmed\Downloads\spider.jpg",
        "feature_vectors.csv"
    )

    print("Most similar images:")
    for path, score in zip(similar_image_paths, similarity_scores):
        print(f"Image Path: {path}, Similarity Score: {score}")
