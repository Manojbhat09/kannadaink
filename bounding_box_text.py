import cv2
import numpy as np

def draw_bounding_boxes_on_text(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to get a binary image
    thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 21, 50)

    # Find contours
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the original image to draw bounding boxes
    bounding_box_image = image.copy()

    # Filter out small contours or draw bounding boxes around large enough contours
    for cnt in contours:
        # Calculate the bounding rectangle
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Optionally, filter out contours that are too small
        if w * h > 100:  # Modify this value based on your use case
            cv2.rectangle(bounding_box_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return bounding_box_image

# Example usage:
image_path = 'kannada.jpg'  # Replace with your image path
output_image_path = 'bounded_image.jpg'  # Replace with your desired save path

# Read the original image
img = cv2.imread(image_path)

# Draw bounding boxes on text
bounded_img = draw_bounding_boxes_on_text(img)

# Save or display the image with bounding boxes
cv2.imwrite(output_image_path, bounded_img)