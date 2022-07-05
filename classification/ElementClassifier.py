from classification.cnn.Data import Data
from classification.cnn.CNN import CNN
from classification.cnn.ImgClassifier import ImgClassifier


class ElementClassifier:
    def __init__(self):
        self.classifier_compo = None
        self.classifier_icon = None
        self.classifier_img = None

    def load_classifiers(self, compo=True, icon=True, img=True):
        if compo:
            data = Data(cls='compo')
            self.classifier_compo = CNN(data)
            self.classifier_compo.load()
        if icon:
            data = Data(cls='icon')
            self.classifier_icon = CNN(data)
            self.classifier_icon.load()
        if img:
            self.classifier_img = ImgClassifier()

    def predict_images(self, images, opt='compo', show=False):
        '''
        :param images: list of cv2 images
        :param opt: Classifier option
            @ 'compo'
            @ 'icon'
            @ 'image'
        :param show: Boolean
        '''
        if opt == 'compo':
            self.classifier_compo.predict_images(images, show)
        elif opt == 'icon':
            self.classifier_icon.predict_images(images, show)
        elif opt == 'image':
            self.classifier_img.predict_images(images, show)

    def predict_img_files(self, img_files, opt='compo', show=False):
        '''
        :param img_files: list of image file paths
        :param opt: Classifier option
            @ 'compo'
            @ 'icon'
            @ 'image'
        :param show: Boolean
        '''
        if opt == 'compo':
            self.classifier_compo.predict_img_files(img_files, show)
        elif opt == 'icon':
            self.classifier_icon.predict_img_files(img_files, show)
        elif opt == 'image':
            self.classifier_img.predict_img_files(img_files, show)


if __name__ == '__main__':
    cls = ElementClassifier()
    cls.load_classifiers()
    cls.predict_img_files(['data/a1.jpg', 'data/a2.jpg'], opt='image', show=True)
