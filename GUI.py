import time
from os.path import join as pjoin
import cv2
import json

from detection.Detector import Detector
from grouping.Grouper import Grouper
from classification.ElementClassifier import ElementClassifier


class GUI:
    def __init__(self, img_file, output_dir='data/output', img_resize_longest_side=800):
        self.img_file = img_file
        self.img = cv2.imread(img_file)
        self.file_name = img_file.replace('\\', '/').split('/')[-1][:-4]
        self.img_reshape = None  # image reshape for element detection
        self.img_resized = None  # resized image by img_reshape
        self.output_dir = output_dir

        self.Detector = Detector(self.img_file, img_resize_longest_side, self.output_dir)            # GUI Element Detection
        self.Grouper = Grouper(self.img_file, self.Detector.detection_result_file, self.output_dir)  # GUI Element Grouping (Layout)
        self.Classifier = ElementClassifier()

        self.detection_result_img = {'text': None, 'non-text': None, 'merge': None}     # visualized detection result
        self.grouping_result_img = {'group': None, 'pair': None, 'list': None}          # visualized detection result

    '''
    *****************************
    *** GUI Element Detection ***
    *****************************
    '''
    def detect_element(self, is_ocr=True, is_non_text=True, is_merge=True, show=True):
        self.Detector.detect_element(is_ocr, is_non_text, is_merge)
        if show:
            self.visualize_element_detection()

    '''
    **********************************
    *** GUI Element Classification ***
    **********************************
    '''
    def classify_element(self):
        pass

    '''
    **************************
    *** Layout Recognition ***
    **************************
    '''
    # entry method
    def recognize_layout(self, show=True):
        # self.Grouper.load_detection_result_from_file()
        self.Grouper.load_compos(self.Detector.compos_json)
        self.Grouper.recognize_layout()
        if show:
            self.visualize_layout_recognition()

    '''
    *********************
    *** Visualization ***
    *********************
    '''
    def visualize_element_detection(self):
        self.Detector.visualize_element_detection()

    def visualize_layout_recognition(self):
        self.Grouper.visualize_layout_recognition()


