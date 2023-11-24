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
                                               cv2.THRESH_BINARY_INV, 21, 30)
    return preprocessed_image

def get_text_bounding_boxes(preprocessed_image):
    # Find contours in the image
    contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Calculate bounding boxes for each contour
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
    return bounding_boxes

def calculate_spacings(bounding_boxes):
    # Calculate vertical distances between bounding box bottoms and tops
    bottoms = [y+h for x, y, w, h in bounding_boxes]
    tops = [y for x, y, w, h in bounding_boxes]
    spacings = np.sort(bottoms) - np.sort(tops)
    return spacings

def analyze_spacings(spacings):
    # Simple approach: consider large gaps as paragraph spacings
    # This threshold could be dynamically determined with more complex analysis
    threshold = np.mean(spacings) + 2 * np.std(spacings)
    line_spacings = spacings[spacings < threshold]
    paragraph_spacings = spacings[spacings >= threshold]
    return line_spacings, paragraph_spacings

def find_largest_cluster(points):
    # Run DBSCAN clustering algorithm
    clustering = DBSCAN(eps=50, min_samples=5).fit(points)
    labels = clustering.labels_

    # Find the largest cluster
    largest_cluster_label = np.bincount(labels[labels >= 0]).argmax()
    largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
    largest_cluster_points = points[largest_cluster_indices]

    # Determine the top-left and bottom-right points of the cluster's bounding box
    min_x, min_y = np.amin(largest_cluster_points, axis=0)
    max_x, max_y = np.amax(largest_cluster_points, axis=0)

    # Calculate the center and extents for the ellipse
    center = (int((min_x + max_x) / 2), int((min_y + max_y) / 2))
    extents = (int((max_x - min_x) / 2), int((max_y - min_y) / 2))

    return center, extents

# Example usage:
image_path = 'kannada.jpg'  # Replace with your image path
output_image_path = 'annotated.jpg'  # Replace with your desired save path

# Read the original image
original_image = cv2.imread(image_path)
processed_image = preprocess_image(original_image)
bounding_boxes = get_text_bounding_boxes(processed_image)
spacings = calculate_spacings(bounding_boxes)
line_spacings, paragraph_spacings = analyze_spacings(spacings)

# Run clustering to find the largest text cluster
central_points = np.array([[x + w / 2, y + h / 2] for x, y, w, h in bounding_boxes])
largest_cluster_center, extents = find_largest_cluster(central_points)

# Draw an ellipse on the original image based on the largest cluster
ellipse_mask = np.zeros_like(processed_image)
# import pdb; pdb.set_trace()
cv2.ellipse(ellipse_mask, largest_cluster_center, extents, 0, 0, 360, 255, -1)
result_image = cv2.bitwise_and(original_image, original_image, mask=ellipse_mask)

# Save the result image
cv2.imwrite(output_image_path, result_image)