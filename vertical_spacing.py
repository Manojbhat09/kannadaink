import cv2
import numpy as np

def compute_vertical_threshold(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to get a binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Calculate the horizontal projection histogram
    hist = cv2.reduce(binary, 1, cv2.REDUCE_AVG).reshape(-1)
    import pdb; pdb.set_trace()

    # Get y-coordinates of text lines (peaks) and spacings (valleys)
    lines = np.where(hist > [np.mean(hist) * 0.5])[0]
    spacings = np.where(hist <= [np.mean(hist) * 0.5])[0]

    # Compute the average spacing between lines
    spacings_diff = np.diff(spacings)
    avg_spacing = np.mean(spacings_diff)

    # Set the vertical threshold to a factor (for example, 1.5) of the average spacing
    vertical_threshold = avg_spacing * 1.5

    return vertical_threshold

def draw_text_lines_and_spacings(image, line_color=(0, 0, 255), spacing_color=(255, 0, 0), line_opacity=0.5, spacing_opacity=0.3):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to get a binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Calculate the horizontal projection histogram
    hist = cv2.reduce(binary, 1, cv2.REDUCE_AVG).reshape(-1)

    # Create a copy of the original image to draw on
    annotated_image = image.copy()

    # Calculate line and spacing colors with opacity
    line_clr = np.array(line_color) * line_opacity
    spacing_clr = np.array(spacing_color) * spacing_opacity

    # Draw semi-transparent lines for text (peaks) and spacings (valleys)
    height, width, _ = image.shape
    for y in range(height):
        if hist[y] > np.mean(hist) * 0.5: # This is a peak (line of text)
            annotated_image[y, :, :] = (1 - line_opacity) * annotated_image[y, :, :] + line_clr
        else: # This is a valley (spacing)
            annotated_image[y, :, :] = (1 - spacing_opacity) * annotated_image[y, :, :] + spacing_clr

    return annotated_image

# Example usage with an image:
image = cv2.imread('kannada.jpg')
annotated_image = draw_text_lines_and_spacings(image)
# vertical_threshold = compute_vertical_threshold(image)

cv2.imwrite('path_to_annotated_image.jpg', annotated_image)