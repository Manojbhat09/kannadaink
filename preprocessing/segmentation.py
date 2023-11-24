import cv2
import os
import ntpath
import numpy as np
from preprocessing.segment_sentence import segment_sentence 
from preprocessing.segment_word import segment_word 
from preprocessing.segment_character import segment_character 

script_path = os.path.abspath(__file__)
ROOT = os.path.dirname(script_path)

class Segmenter:
    def __init__(self, input_data):
        if isinstance(input_data, str):  # input is a file path
            self.image = cv2.imread(input_data)
        elif isinstance(input_data, np.ndarray):  # input is a numpy array
            self.image = input_data
        else:
            raise ValueError("Invalid input: input_data should be either a file path or a numpy array")
        self.rootdir = os.path.join(ROOT, "data", "processed")

    def segment(self):
        sentences = self.segment_sentence_()
        words, word_characters_list, word_ottaksharas_list = [], [], []
        for i, sentence in enumerate(sentences):
            sentence_words = self.segment_word_(sentence, i)
            words.extend(sentence_words)
            for j, word in enumerate(sentence_words):
                word_characters, word_ottaksharas = self.segment_character_(word)
                word_characters_list.append(word_characters)
                word_ottaksharas_list.extend(word_ottaksharas)
        return sentences, words, word_characters_list, word_ottaksharas_list

    def segment_sentence_(self):
        directory = self.rootdir + "/lines"
        if not os.path.exists(directory):
            os.makedirs(directory)
        sentences = segment_sentence(self.image, directory)
        return sentences

    def segment_word_(self, image, count):
        directory = self.rootdir + "/words"
        if not os.path.exists(directory):
            os.makedirs(directory)

        words = segment_word(image, directory, count)
        return words

    def segment_character_(self, image):
        directory = self.rootdir
        characters, ottaksharas = segment_character(image, directory)
        return characters, ottaksharas

    @staticmethod
    def sort_contours(cnts, method="left-to-right"):
        # initialize the reverse flag and sort index
        reverse = False
        i = 0

        # handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True

        # handle if we are sorting against the y-coordinate rather than
        # the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1

        # construct the list of bounding boxes and sort them from top to bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))

        # return the list of sorted contours and bounding boxes
        return (cnts, boundingBoxes)


# Usage:
# just segmenting the text meaning to seggregate the words characters and letters from the patch of the sentence and the lines of text
# after the text line detection the next step is to crop the patches and segment the character, which is what 
# this script is doing. After which single and multiple character patches are sent to the classifier to 
# classify to the output classes and unicode text through mapping 
# at the output the intention is to get the kannada symbols along with the corresonding ottaksharaas 
# the output is simplified on top with an LLM doing the trans literation into english and mainting the senbility and sturecture of the scentence s
# So the basic prcoess is from images to transcript in different languages 
if __name__ == "__main__":
    segmenter = Segmenter(os.path.join(ROOT, '../data/kannada.jpg'))
    sentences, words, characters, ottaksharas = segmenter.segment()
    import pdb; pdb.set_trace()

