import itertools
from tensorflow.keras.utils import Sequence, to_categorical
from PIL import Image
import numpy as np


class Generator(Sequence):
    def __init__(self, data_path, class_path, height, width, n_class, batch_size) -> None:
        self.data_path = data_path
        self.class_path = class_path
        self.heigth = height
        self.width = width
        self.n_class = n_class
        self.batch_size = batch_size
        self.length = len(data_path)
        self.batches_per_epoch = int((self.length - 1) / batch_size) + 1

    def __color2gray__(self, label):
        label[label == 11] = 1
        label[label == 76] = 2
        label[label == 104] = 3
        label[label == 211] = 4
        label[label == 226] = 5
        label[label == 141] = 6
        label[label == 97] = 7
        label[label == 164] = 8
        label[label == 80] = 9

        return label

    def __getitem__(self, idx):
        start_pos = self.batch_size*idx
        end_pos = start_pos+self.batch_size

        if end_pos > self.length:
            end_pos = self.length

        image_paths = self.data_path[start_pos:end_pos]
        class_paths = self.class_path[start_pos:end_pos]

        images = []
        labels = []
        for i, (image_path, class_path) in enumerate(zip(image_paths, class_paths)):
            img = np.array(Image.open(image_path).convert('RGB'))/255.0
            label = np.array(Image.open(class_path).convert('L'))
            label = self.__color2gray__(label)
            h, w = img.shape[0], img.shape[1]
            for x, y in itertools.product(range(0, w-self.width, int(self.width)), range(0, h-self.heigth, int(self.heigth))):
                images.append(img[y:y+self.heigth, x:x+self.width])
                labels.append(label[y:y+self.heigth, x:x+self.width])

        return np.array(images), to_categorical(np.array(labels), self.n_class, dtype='int8')

    def __len__(self):
        return self.batches_per_epoch

    def on_epoch_end(self):
        pass
