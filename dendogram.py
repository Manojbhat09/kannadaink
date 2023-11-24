import numpy as np
import cv2
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

def preprocess_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Threshold the image
    _, binary = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary

def get_text_line_positions(binary_image):
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Determine the vertical center positions of text lines
    vertical_centers = [cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] // 2 for c in contours]
    # Reshape for hierarchical clustering
    vertical_centers = np.array(vertical_centers).reshape(-1, 1)
    return vertical_centers

def perform_hierarchical_clustering(data):
    # Perform the hierarchical clustering
    Z = linkage(data, 'ward')
    # Plot the dendrogram
    plt.figure(figsize=(10, 5))
    dendrogram(Z)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Index')
    plt.ylabel('Distance')
    plt.savefig('clustered_image2.jpg')
    return Z

# Annotate and save original image with clusters
def annotate_and_save_image(image, line_positions, clusters):
    annotated_image = image.copy()
    for k in range(min(clusters), max(clusters) + 1):
        cluster_indices = np.where(clusters == k)[0]
        for i in cluster_indices:
            y_val = line_positions[i][0]
            cv2.line(annotated_image, (0, y_val), (image.shape[1], y_val), (0, 255, 0), 2)
    
    # Save annotated image with cluster lines
    cv2.imwrite('clustered_image.jpg', annotated_image)


# Main processing
image_path = 'kannada.jpg'
output_dendrogram_path = 'dendrogram.png'
output_clustered_image_path = 'clustered_image2.jpg'

# Replace the paths above with the actual paths on your system

# Load the image and preprocess
image = cv2.imread(image_path)
binary_image = preprocess_image(image)

# Extract the positions of text lines
line_positions = get_text_line_positions(binary_image)

# Perform Hierarchical Clustering
linkage_matrix = perform_hierarchical_clustering(line_positions)

# Choose a cut-off distance (based on dendrogram inspection)
cut_off_distance = 1000  # Adjust this cutoff based on your dendrogram outcome

# Create clusters based on the cut-off distance
clusters = fcluster(linkage_matrix, cut_off_distance, criterion='distance')

# Annotate and save the image with text line clusters marked on it
annotate_and_save_image(image, line_positions, clusters)