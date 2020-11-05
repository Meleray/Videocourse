import os
import numpy as np
from skimage.io import imread
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.transform import resize

ROT_ANGLE = 30
BATCH_SIZE = 256
IMG_SIZE = 200
EPOCHS = 100


def train_classifier(train_gt, train_img_dir, fast_train=True):
    if fast_train:
        len_train = 4
        batches = 2
        epochs = 1
    else:
        len_train = len(train_gt)
        batches = BATCH_SIZE
        epochs = EPOCHS
    X = np.zeros((len_train, IMG_SIZE, IMG_SIZE, 3))
    y = np.zeros(len_train)
    i = 0
    for filename in train_gt:
        img = resize(imread(os.path.join(train_img_dir, filename)), (IMG_SIZE, IMG_SIZE, 3))
        X[i] = img
        y[i] = train_gt[filename]
        i += 1
        if i == len_train:
            break
    datagen = ImageDataGenerator(rotation_range = ROT_ANGLE, horizontal_flip = True, width_shift_range=0.05, height_shift_range=0.05, validation_split=0.15)
    clf = tf.keras.applications.Xception()
    clf.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    clf.fit(datagen.flow(X, y, batch_size=batches), steps_per_epoch = (int)(np.ceil(len_train / batches)), epochs = epochs)
    #clf.save('/content/drive/My Drive/Colab Notebooks/bird_classification/facepoints_model.hdf5')

def classify(clf, test_img_dir):
    filenames = os.listdir(test_img_dir)
    len_test = len(filenames)
    X_test = np.zeros((len_test, IMG_SIZE, IMG_SIZE, 3))
    for i in range(len(filenames)):
        filename = filenames[i]
        X_test[i] = resize(imread(os.path.join(test_img_dir, filename)), (IMG_SIZE, IMG_SIZE, 3))
    return clf.predict(X_test)
