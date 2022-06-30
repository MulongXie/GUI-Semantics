import tensorflow.keras as keras
from keras_applications.resnet50 import ResNet50
from keras.models import Model,load_model
from keras.layers import Dense, Activation, Flatten, Dropout
from sklearn.metrics import confusion_matrix
import numpy as np
import cv2


class CNN:
    def __init__(self, data):
        self.data = data    # Data object
        self.model = None
        self.training_history = None

        self.image_shape = data.image_shape
        self.class_map = data.class_map
        self.class_number = data.class_number
        self.model_path = '/home/ml/Model/Rico/component/component1.h5'

    def build_model(self, epoch_num):
        base_model = ResNet50(include_top=False, weights='imagenet', input_shape=self.image_shape,
                              backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
        for layer in base_model.layers:
            layer.trainable = False
        self.model = Flatten()(base_model.output)
        self.model = Dense(128, activation='relu')(self.model)
        self.model = Dropout(0.5)(self.model)
        self.model = Dense(self.class_number, activation='softmax')(self.model)
        self.model = Model(inputs=base_model.input, outputs=self.model)

    def train(self, epoch_num=30, continue_with_loading=False):
        if continue_with_loading:
            self.load()
        else:
            self.build_model(epoch_num)
        self.model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        self.training_history = self.model.fit(self.data.X_train, self.data.Y_train, batch_size=64, epochs=epoch_num, verbose=1, validation_data=(self.data.X_test, self.data.Y_test))
        self.model.save(self.model_path)
        print("Trained model is saved to", self.model_path)

    def load(self):
        self.model = load_model(self.model_path)
        print('Model Loaded From', self.model_path)

    def preprocess_img(self, image):
        image = cv2.resize(image, self.image_shape[:2])
        x = (image / 255).astype('float32')
        x = np.array([x])
        return x

    def predict(self, imgs, show=False):
        if self.model is None:
            print("*** No model loaded ***")
            return
        for i in range(len(imgs)):
            X = self.preprocess_img(imgs[i])
            Y = self.class_map[np.argmax(self.model.predict(X))]
            if show:
                print(Y)
                cv2.imshow('element', imgs[i])
                cv2.waitKey()

    def evaluate(self, data):
        X_test = data.X_test
        Y_test = [np.argmax(y) for y in data.Y_test]
        Y_pre = [np.argmax(y_pre) for y_pre in self.model.predict(X_test, verbose=1)]

        matrix = confusion_matrix(Y_test, Y_pre)
        print(matrix)
        TP, FP, FN = 0, 0, 0
        for i in range(len(matrix)):
            TP += matrix[i][i]
            FP += sum(matrix[i][:]) - matrix[i][i]
            FN += sum(matrix[:][i]) - matrix[i][i]
        precision = TP/(TP+FP)
        recall = TP / (TP+FN)
        print("Precision:%.3f, Recall:%.3f" % (precision, recall))