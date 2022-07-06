import time
from os.path import join as pjoin
import cv2
import json

from detection.Detector import Detector
from grouping.Grouper import Grouper
from classification.Classifier import Classifier
from Element import Element


class GUI:
    def __init__(self, img_file, output_dir='data/output', img_resize_longest_side=800):
        self.img_file = img_file
        self.img = cv2.imread(img_file)
        self.file_name = img_file.replace('\\', '/').split('/')[-1][:-4]
        self.output_dir = output_dir

        self.Detector = Detector(self.img_file, img_resize_longest_side, self.output_dir)            # GUI Element Detection
        self.Grouper = Grouper(self.img_file, self.Detector.detection_result_file, self.output_dir)  # GUI Element Grouping (Layout)
        self.Classifier = Classifier()

        self.img_reshape = self.Detector.img_reshape  # image reshape for element detection
        self.img_resized = self.Detector.img_resized  # resized image by img_reshape

        self.elements = []  # list of Element objects
        self.color_map = {'Compo': (0,255,0), 'Text':(0,0,255)}

        self.detection_result_img = {'text': None, 'non-text': None, 'merge': None}     # visualized detection result
        self.grouping_result_img = {'group': None, 'pair': None, 'list': None}          # visualized detection result

    '''
    *****************************
    *** GUI Element Detection ***
    *****************************
    '''
    def detect_element(self, is_ocr=True, is_non_text=True, is_merge=True, show=True):
        self.Detector.detect_element(is_ocr, is_non_text, is_merge)
        self.cvt_elements()
        if show:
            self.visualize_element_detection()

    def load_detection_result(self):
        self.Detector.load_detection_result()
        self.cvt_elements()

    def cvt_elements(self):
        '''
        Convert detection result to Element objects
        '''
        compos = self.Detector.compos_json['compos']
        for compo in compos:
            pos = compo['position']
            element = Element(compo['id'], pos['column_min'], pos['row_min'], pos['column_max'], pos['row_max'])
            element.attributes.element_class = compo['class']
            element.get_clip(self.img_resized)
            self.elements.append(element)

    '''
    **********************************
    *** GUI Element Classification ***
    **********************************
    '''
    def classify_element(self):
        self.Classifier.load_classifiers()
        # separate text and non-text compos
        compos = []
        texts = []
        for ele in self.elements:
            if ele.attributes.element_class == 'Compo':
                compos.append(ele)
            elif ele.attributes.element_class == 'Text':
                texts.append(ele)
        # 1. classify compo class
        compos_clips = [compo.clip for compo in compos]
        labels = self.Classifier.predict_images(compos_clips, opt='compo')
        for i, compo in enumerate(compos):
            compo.attribute.compo_class = labels[i]

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

    def draw_elements(self):
        board = self.img_resized.copy()
        for element in self.elements:
            element.draw_element(board, self.color_map[element.attributes.element_class])
        cv2.imshow('elements', board)
        cv2.waitKey()
        cv2.destroyWindow('elements')
