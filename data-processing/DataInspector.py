import json
from glob import glob
from os.path import join as pjoin
import cv2
from GUIData import GUIData


class DataInspector:
    def __init__(self, data_directory='E:\\Mulong\\Datasets\\gui\\rico\\combined\\all'):
        # the directory to store json file and image file
        self.data_directory = data_directory

        # files' paths, list of string
        self.img_files = None
        self.json_files = None

        self.guis = []  # list of GUIData objects
        self.gui_id = 0

    def get_all_img_files_on_data_directory(self, img_type='.jpg'):
        self.img_files = glob(pjoin(self.data_directory, '*' + img_type))

    def get_all_json_files_on_data_directory(self):
        self.json_files = glob(pjoin(self.data_directory, '*.json'))

    def generate_file_path(self, file_name, file_type='.jpg'):
        name = file_name.replace('/', '\\').split('\\')[-1].split('.')[0]
        target_file = pjoin(self.data_directory, name + file_type)
        return target_file

    def load_gui(self, image_file, json_file):
        gui = GUIData(self.gui_id, image_file, json_file)
        self.gui_id += 1
        self.guis.append(gui)
        return gui

    def inspect_gui_img_and_element(self, img_file, json_file, extract_from='vh', save_as_df=False):
        '''
        Inspect GUI by visualizing the image and printing out the json file of attributes
        :param img_file: GUI image file path
        :param json_file: GUI Json data file path
        :param extract_from:
            @'vh' - from raw view hierarchy file (start with 'activity')
            @'semantic' - from the semantic json file processed by Rico
        :param save_as_df: Boolean, if True, save the elements into csv where the row title is attribute
        '''
        gui = self.load_gui(img_file, json_file)
        if extract_from == 'vh':
            gui.extract_elements_from_vh()
        else:
            gui.extract_element_from_semantic_tree()
        gui.visualize_elements()
        if save_as_df:
            gui.save_element_as_csv()


if __name__ == '__main__':
    data = DataInspector()
    data.get_all_json_files_on_data_directory()

    for jfile in data.json_files:
        print('***********', jfile)
        imgfile = data.generate_file_path(jfile)
        data.inspect_gui_img_and_element(imgfile, jfile)  # press 'q' to exit
