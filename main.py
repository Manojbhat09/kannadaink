import cv2
import pandas as pd
from preprocessing.segmentation import Segmenter  # This is from your provided code
# from denoising_model import denoise  # assuming you have a module for denoising
# assuming a module for bounding box detection

from models.classifier import CharacterClassifier  # assuming a module for character classification
from models.bounding_box_detection import BoundingBoxDetector 
from models.classifier import CharacterClassifier
import torch
import os
# Load the mapping from class names to unicode characters
CHECKPOINTS_PATH = ""
class_to_unicode = pd.read_csv('online_mapping.csv').set_index('Folder').to_dict()
common_unicode_names = class_to_unicode['Most Common Unicode'].keys()
common_unicode_text = class_to_unicode['Most Common Unicode'].values()

def denoise(image):
    # Convert image to grayscale if it's colored
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    # Apply Non-Local Means Denoising
    denoised_image = cv2.fastNlMeansDenoising(gray_image, None, 30, 7, 21)
    return denoised_image

def group_boxes_into_lines(boxes, vertical_threshold):
    """Group bounding boxes that are within a vertical threshold of each other.
    
    Args:
        boxes (list of lists): List of bounding box coordinates [xmin, ymin, xmax, ymax].
        vertical_threshold (int): The vertical distance threshold to consider boxes on the same line.
        
    Returns:
        list of lists: A list where each sublist represents a group of boxes in the same line.
    """
    
    # Step 1: Start with an empty list for lines
    lines = []
    
    # Step 2: Iterate over each box to assign it to a line
    for box in boxes:
        placed = False
        ymin, ymax = box[1], box[3]
        
        # Step 3: Compare with existing lines to find a match
        for line in lines:
            # Assume line is a list of boxes, and we're taking an average Y position from the first box
            # Modify this logic if boxes in a line can have significant height differences
            line_ymin = line[0][1]
            line_ymax = line[0][3]
            line_yaverage = (line_ymin + line_ymax) / 2
            
            # Step 4: Check if the current box fits within this line based on vertical_threshold 
            if ymin <= line_yaverage + vertical_threshold and ymax >= line_yaverage - vertical_threshold:
                # Add box to the line as it fits within the threshold
                line.append(box)
                placed = True
                break
        
        # Step 5: If the box wasn't placed in any existing line, create a new line
        if not placed:
            lines.append([box])
            
    # Step 6: Return the grouped lines
    return lines


def process_image(file_path):
    # Step 1: Denoise the image
    raw_image = cv2.imread(file_path)
    # import pdb; pdb.set_trace()
    denoised_image = denoise(raw_image)
    
    # Step detector_checkpoint: Get bounding box coordinates
    detector_checkpoint = os.path.join(CHECKPOINTS_PATH, 'yolov8m_train10_last.pt')
    bbox_detector = BoundingBoxDetector(detector_checkpoint)

    # Assume bbox_detector.get_bounding_boxes(denoised_image) returns all bounding boxes from image
    bounding_boxes = bbox_detector.get_bounding_boxes(denoised_image)

    # Sort boxes by vertical position (top-down)
    sorted_boxes = sorted(bounding_boxes[0], key=lambda box: box[1])  # Sorting by ymin value

    # Group boxes into lines
    lines = group_boxes_into_lines(sorted_boxes, vertical_threshold)

    # Initialize the content for the output file
    output_content = []

    # Process each line
    for line in lines:
        # Sort the line's boxes from left to right (reading order)
        sorted_line = sorted(line, key=lambda box: box[0])  # Sorting by xmin value

        line_string = ''
        
        for box in sorted_line:
            xmin, ymin, xmax, ymax = box.astype(int)
            cropped_image = denoised_image[ymin:ymax, xmin:xmax]
            
            # Segment the cropped image to get characters
            segmenter = Segmenter(cropped_image)
            _, _, characters, _ = segmenter.segment()
            
            if characters:
                classifier_checkpoint = os.path.join(CHECKPOINTS_PATH, 'classifier.pth')
                classifier = CharacterClassifier(classifier_checkpoint)

                for char_img in list(characters[0].values()):
                    # Further processing such as resizing, classification, etc.
                    unicode_character = ...  # Obtain the unicode character from the classifier
                    line_string += unicode_character

        # Add the line string to the output content
        output_content.append(line_string)

    # Write output content to the file
    with open('transliteration.txt', 'w', encoding='utf-8') as output_file:
        for line in output_content:
            output_file.write(line + '\n')
    
    for box in bounding_boxes[0]:
        
        xmin, ymin, xmax, ymax = box.astype(int)
        cropped_image = denoised_image[ymin:ymax, xmin:xmax]
        
        # Step 3: Segment the cropped image
        segmenter = Segmenter(cropped_image)
        sentences, words, characters, ottaksharas = segmenter.segment()
        classifier_checkpoint = os.path.join(CHECKPOINTS_PATH, 'classifier.pth')
        classifier = CharacterClassifier(classifier_checkpoint)
        sentence_string = ""
        if not characters:
            print("No characters found")
            continue
        for char_img in list(characters[0].values()):
            char_img = cv2.resize(char_img, (max(100, char_img.shape[1]), max(100, char_img.shape[0])), interpolation=cv2.INTER_AREA)

            if char_img.shape[0] < 10 or char_img.shape[1] < 10:  # Check dimensions
                with open('transliteration.txt', 'a', encoding='utf-8') as file:
                    file.write(sentence_string + '\n')
                sentence_string = ""  # Reset sentence_string
                print("skipping")
                continue  
            # Step 4: Classify each character
            char_img_tensor = torch.Tensor(char_img).view(1, 1, char_img.shape[0], char_img.shape[1])
            class_label = classifier.classify(char_img_tensor)
            common_unicode_text_list = list(common_unicode_text)
            unicode_char = common_unicode_text_list[class_label[0]]
            unicode_char = unicode_char.replace("'", "")
            sentence_string += unicode_char
            # Step 5: Map to Unicode character
            print(unicode_char)

        # import pdb; pdb.set_trace()
        # Write the sentence to the file
        with open('transliteration.txt', 'a', encoding='utf-8') as file:
            file.write(sentence_string + '\n')

# Call the function
process_image('kannada.jpg')

# add in the process to convert the work fthe main.[py] should have more than
