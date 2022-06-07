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

    def get_img_files(self):
        self.img_files = glob(pjoin(self.data_directory, '.jpg'))

    def get_json_files(self):
        self.json_files = glob(pjoin(self.data_directory, '*.json'))

    def get_another_file(self, file_name, target_type='.jpg'):
        name = file_name.replace('/', '\\').split('\\')[-1].split('.')[0]
        target_file = pjoin(self.data_directory, name + target_type)
        return target_file

    def load_gui(self, image_file, json_file):
        gui = GUIData(self.gui_id, image_file, json_file)
        self.guis.append(gui)
        return gui

