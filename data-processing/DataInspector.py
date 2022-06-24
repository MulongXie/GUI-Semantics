import json
from glob import glob
from os.path import join as pjoin
import cv2
from GUIData import GUIData


class DataInspector:
    def __init__(self, img_directory, json_directory):
        # the directory to store json file and image file
        self.img_directory = img_directory
        self.json_directory = json_directory

        # files' paths, list of string
        self.img_files = None
        self.json_files = None

        self.guis = []  # list of GUIData objects
        self.element_clips_count = {}  # {'componentLabel': No_clips}

    def get_all_img_files(self, img_type='.jpg', sort_files_by_name=True):
        self.img_files = glob(pjoin(self.img_directory, '*' + img_type))
        if sort_files_by_name:
            self.img_files = sorted(self.img_files, key=lambda x: int(x.replace('\\', '/').split('/')[-1].split('.')[0]))

    def get_all_json_files(self, sort_files_by_name=True):
        self.json_files = glob(pjoin(self.json_directory, '*.json'))
        if sort_files_by_name:
            self.json_files = sorted(self.json_files, key=lambda x: int(x.replace('\\', '/').split('/')[-1].split('.')[0]))

    def generate_file_path(self, source_file, target_file_type='.jpg'):
        '''
        Given a source JSON file, generate the corresponding image file path or Vice Versa
        :param source_file:  the given source file, to get the file name
        :param target_file_type: the target file type to generate
        :return: the generated file path
        '''
        name = source_file.replace('/', '\\').split('\\')[-1].split('.')[0]
        target_file = None
        # generate image path
        if target_file_type in ('.jpg', '.png'):
            target_file = pjoin(self.img_directory, name + target_file_type)
        # generate json path
        elif target_file_type == '.json':
            target_file = pjoin(self.json_directory, name + target_file_type)
        return target_file

    def load_gui_img_and_element(self, img_file, json_file, show=False, save_as_df=False, save_element_clips=False, output_compo_dir=None):
        '''
        Inspect GUI by visualizing the image and printing out the json file of attributes
        '''
        gui = GUIData(img_file, json_file)
        self.guis.append(gui)

        gui.extract_element_from_semantic_tree()
        if show:
            gui.visualize_elements()
        if save_as_df:
            gui.save_element_as_csv()
        if save_element_clips:
            gui.save_elements_clips_by_compo_label(output_compo_dir, self.element_clips_count)
