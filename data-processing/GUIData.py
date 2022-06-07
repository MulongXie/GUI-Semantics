import cv2
import json


class GUIData:
    def __init__(self, gui_id, gui_img_file, gui_vh_file):
        self.id = gui_id

        self.img_file = gui_img_file
        self.vh_file = gui_vh_file

        self.img = cv2.resize(cv2.imread(gui_img_file), (1440, 2560))  # cv2 image, the screenshot of the GUI
        self.vh = json.load(open(gui_vh_file, 'r'))  # json data, the view hierarchy of the GUI

        self.elements = []      # list of element in dictionary {'id':, 'class':...}
        self.element_id = 0

    def extract_elements_from_vh(self):
        '''
        Extract elements from vh and store them as dictionaries
        '''
        element_root = self.vh['activity']['root']
        self.extract_children_elements(element_root)

    def extract_children_elements(self, element):
        '''
        Recursively extract children from an element
        '''
        element['id'] = self.element_id
        self.elements.append(element)
        if 'children' in element:
            element['children_id'] = []
            for child in element['children']:
                self.element_id += 1
                element['children_id'].append(self.element_id)
                self.extract_children_elements(child)
            # replace wordy 'children' with 'children_id'
            del element['children']
        del element['ancestors']
