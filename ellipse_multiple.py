from sklearn.cluster import DBSCAN
import numpy as np
import cv2

def preprocess_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply adaptive thresholding to prepare for contour detection
    preprocessed_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY_INV, 21, 20)
    return preprocessed_image

def get_text_bounding_boxes(preprocessed_image):
    # Find contours in the image
    contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Calculate bounding boxes for each contour
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
    return bounding_boxes

def find_ellipses_for_clusters(points):
    # Run DBSCAN clustering algorithm
    clustering = DBSCAN(eps=70, min_samples=5).fit(points)
    labels = clustering.labels_
    
    ellipses = []  # To store the ellipses for each cluster

    # Iterate through each cluster label except for noise (label == -1)
    for label in np.unique(labels[labels >= 0]):
        # Find points in this cluster
        cluster_points = points[labels == label]

        # Compute the minimum enclosing ellipse
        if cluster_points.shape[0] >= 5:  # cv2.fitEllipse requires at least 5 points
            formatted_points = np.array(cluster_points, dtype=np.float32)
            ellipse = cv2.fitEllipse(np.array(formatted_points))
            ellipses.append(ellipse)
    
    return ellipses

# Main processing pipeline
image_path = 'kannada.jpg'  # Replace with your image path
original_image = cv2.imread(image_path)
processed_image = preprocess_image(original_image)
bounding_boxes = get_text_bounding_boxes(processed_image)

# Central points of bounding boxes (x+w/2, y+h/2)
central_points = np.array([[x + w / 2, y + h / 2] for x, y, w, h in bounding_boxes])

# Find all possible ellipses in the image covering text clusters
ellipses = find_ellipses_for_clusters(central_points)

# Draw these ellipses on the image
annotated_image = original_image.copy()
for ellipse in ellipses:
    cv2.ellipse(annotated_image, ellipse, (0, 255, 0), 2)

# Save or display the image with ellipses
output_image_path = 'multiple_ellipse.jpg'  # Replace with your desired save path
cv2.imwrite(output_image_path, annotated_image)