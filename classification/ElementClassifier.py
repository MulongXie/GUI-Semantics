from cnn.Data import Data
from cnn.CNN import CNN
from cnn.ImgClassifier import ImgClassifier


class ElementClassifier:
    def __init__(self):
        self.classifier_compo = None
        self.classifier_icon = None
        self.classifier_img = None

    def load_classifiers(self, compo=True, icon=True, img=True):
        if compo:
            data = Data(cls='compo')
            self.classifier_compo = CNN(data)
        if icon:
            data = Data(cls='icon')
            self.classifier_icon = CNN(data)
        if img:
            self.classifier_img = ImgClassifier()

    def predict(self, images, opt='compo', show=False):
        '''
        :param show: Boolean
        :param images: list of cv2 images
        :param opt: Classifier option
            @ 'compo'
            @ 'icon'
            @ 'image'
        '''
        if opt == 'compo':
            self.classifier_compo.predict_images(images, show)
        elif opt == 'icon':
            self.classifier_icon.predict_images(images, show)
        elif opt == 'image':
            self.classifier_img.predict_images(images, show)
