import cv2
import numpy as np

# Function to detect bounding boxes around text regions in an image
def get_text_bounding_boxes(gray):
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to get a binary image
    thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 21, 50)

    # Find contours
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Collect bounding boxes for contours of significant size
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 100]

    return bounding_boxes

# Function to draw an ellipse from bounding boxes
def draw_ellipse_from_bounding_boxes(image, bounding_boxes):
    # Calculate the center of the image
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
    
    # Get the extent of text coverage horizontally and vertically
    max_x, max_y = max([x + w for x, y, w, h in bounding_boxes]), max([y + h for x, y, w, h in bounding_boxes])
    min_x, min_y = min([x for x, y, w, h in bounding_boxes]), min([y for x, y, w, h in bounding_boxes])
    
    # Calculate extents of the ellipse
    width_extent = (max_x - min_x) // 2
    height_extent = (max_y - min_y) // 2

    # Calculate the center point of the bounding boxes
    boxes_center_x, boxes_center_y = (max_x + min_x) // 2, (max_y + min_y) // 2
    
    # Create a mask with an ellipse
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.ellipse(mask, (boxes_center_x, boxes_center_y), (width_extent, height_extent), 0, 0, 360, 255, -1)

    # Bitwise AND to keep only the regions inside the ellipse
    cropped_image = cv2.bitwise_and(image, image, mask=mask)

    return cropped_image

# Example usage:
image_path = 'kannada.jpg'  # Replace with your image path
output_image_path = 'ellipse_cropped_image.jpg'  # Replace with your desired save path

# Read the original image and convert to grayscale
original_image = cv2.imread(image_path)
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Get text bounding boxes
text_bounding_boxes = get_text_bounding_boxes(gray_image)

# Draw an ellipse from the text bounding boxes
ellipse_cropped_img = draw_ellipse_from_bounding_boxes(original_image, text_bounding_boxes)

# Save or display the ellipse cropped image
cv2.imwrite(output_image_path, ellipse_cropped_img)