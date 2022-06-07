import cv2
import json
import pandas as pd


class GUIData:
    def __init__(self, gui_id, gui_img_file, gui_vh_file):
        self.id = gui_id

        self.img_file = gui_img_file
        self.vh_file = gui_vh_file

        self.img = cv2.resize(cv2.imread(gui_img_file), (1440, 2560))  # cv2 image, the screenshot of the GUI
        self.vh = json.load(open(gui_vh_file, 'r'))  # json data, the view hierarchy of the GUI

        self.elements = []       # list of element in dictionary {'id':, 'class':...}
        self.element_id = 0
        self.elements_df = None  # pandas.dataframe

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
        # discard illegal elements
        if not (element['bounds'][0] == element['bounds'][2] or element['bounds'][1] == element['bounds'][3]) and \
                not ('layout' in element['class'].lower()):
            self.elements.append(element)
        if 'children' in element:
            element['children-id'] = []
            for child in element['children']:
                self.element_id += 1
                element['children-id'].append(self.element_id)
                self.extract_children_elements(child)
            # replace wordy 'children' with 'children-id'
            del element['children']
        del element['ancestors']

    def cvt_elements_to_dataframe(self):
        self.elements_df = pd.DataFrame(self.elements)

    def save_element_as_csv(self, file_name='elements.csv'):
        if self.elements_df is None:
            self.cvt_elements_to_dataframe()
        self.elements_df.to_csv(file_name)

    def visualize_elements(self):
        board = self.img.copy()
        for ele in self.elements:
            print(ele['id'], ele['class'])
            print(ele, '\n')
            bounds = ele['bounds']
            clip = self.img[bounds[1]: bounds[3], bounds[0]: bounds[2]]
            cv2.rectangle(board, (bounds[0], bounds[1]), (bounds[2], bounds[3]), (0, 255, 0), 3)
            cv2.imshow('clip', cv2.resize(clip, (clip.shape[1] // 3, clip.shape[0] // 3)))
            cv2.imshow('ele', cv2.resize(board, (board.shape[1] // 3, board.shape[0] // 3)))
            if cv2.waitKey() == ord('q'):
                break
        cv2.destroyAllWindows()
