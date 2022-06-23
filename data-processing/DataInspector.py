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

    def get_all_img_files_on_data_directory(self):
        self.img_files = glob(pjoin(self.data_directory, '.jpg'))

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


if __name__ == '__main__':
    data = DataInspector()
    data.get_all_json_files_on_data_directory()

    for jfile in data.json_files:
        print('***********', jfile)
        imgfile = data.generate_file_path(jfile)

        gui = data.load_gui(imgfile, jfile)
        gui.extract_elements_from_vh()
        gui.visualize_elements()  # press 'q' to exit
        # gui.save_element_as_csv()
        # break
