import os
import numpy as np
from skimage.io import imread
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization, Reshape)
from skimage.transform import resize, rotate
from skimage.color import rgb2gray


IMG_SIZE = 100
EPOCHS = 300
BATCHES = 256


def flip_img(img, pnts):
    new_pnts = np.copy(pnts)
    new_pnts[:, 0] = -pnts[:, 0] + np.shape(img)[1] - 1
    new_pnts[0], new_pnts[3] = new_pnts[3], new_pnts[0]
    new_pnts[1], new_pnts[2] = new_pnts[2], new_pnts[1]
    new_pnts[4], new_pnts[9] = new_pnts[9], new_pnts[4]
    new_pnts[5], new_pnts[8] = new_pnts[8], new_pnts[5]
    new_pnts[6], new_pnts[7] = new_pnts[7], new_pnts[6]
    new_pnts[11], new_pnts[13] = new_pnts[13], new_pnts[11]
    return (np.fliplr(img), new_pnts)

def random_rotate(img, pnts):
    angle = np.random.randint(-30, 30)
    rotation_matrix = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    center = np.shape(img)[0] / 2 - 0.5
    new_pnts = np.matrix.transpose(np.matmul(rotation_matrix, np.matrix.transpose(pnts - center))) +  center
    return (rotate(img, angle), new_pnts)

def resize_img(img, pnts, ret_reverse=False):
    new_img = resize(img, [IMG_SIZE, IMG_SIZE])
    if pnts is not None:
        new_pnts = pnts * IMG_SIZE / np.shape(img)[0]
    else:
        new_pnts = None
    if ret_reverse:
        return (new_img, new_pnts, np.shape(img)[0] / IMG_SIZE)
    else:
        return (new_img, new_pnts)

def random_crop(img, pnts):
    crop = (int)((np.random.randint(0, 30) / 1000.0) * IMG_SIZE)
    return resize_img(img[crop:IMG_SIZE - crop:, crop:IMG_SIZE - crop:], pnts - crop, False)

def normalize_img(img):
    avg = np.mean(img, axis=0)
    disp = np.sqrt(np.var(img, axis=0))
    return np.array((img - avg) / disp).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

def get_train_data(img_train, dir, fast_train=False):
    if fast_train:
        len_train = 3
    else:
        len_train = len(img_train) * 7
    X = np.zeros((len_train, IMG_SIZE, IMG_SIZE))
    y = np.zeros((len_train, 14, 2))
    i = 0
    for filename in img_train:
        #if i % 700 == 0:
          #print(i)
        pnts = img_train[filename]
        img = rgb2gray(imread(os.path.join(dir, filename)))
        new_pnts = np.dstack((pnts[::2], pnts[1::2]))[0]
        img, new_pnts = resize_img(img, new_pnts, False)
        X[i] = img
        y[i] = new_pnts
        if fast_train:
            i += 1
            if i == len_train:
                break
        else:
            X[i + 6], y[i + 6] = random_crop(img, new_pnts)
            X[i + 5], y[i + 5] = flip_img(img, new_pnts)
            for j in range(4):
                X[i + j + 1], y[i + j + 1] = random_rotate(img, new_pnts)
            i += 7
    return normalize_img(X), y


def get_test_data(dir):
    filenames = os.listdir(dir)
    len_test = len(filenames)
    X = np.zeros((len_test, IMG_SIZE, IMG_SIZE))
    transforms = np.zeros(len_test)
    for i in range(len(filenames)):
        filename = filenames[i]
        img = rgb2gray(imread(os.path.join(dir, filename)))
        img, tmp, transform = resize_img(img, None, True)
        X[i] = img
        transforms[i] = transform
    return normalize_img(X), filenames, transforms


def build_clf():
    clf = Sequential()

    clf.add(Conv2D(32, (4, 4), padding='valid', kernel_initializer='random_uniform', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    clf.add(Activation('elu'))
    clf.add(MaxPooling2D((2, 2)))
    clf.add(Dropout(0.1))

    clf.add(Conv2D(64, (3, 3), padding='valid', kernel_initializer='random_uniform'))
    clf.add(Activation('elu'))
    clf.add(MaxPooling2D((2, 2)))
    clf.add(Dropout(0.2))

    clf.add(Conv2D(128, (2, 2), padding='valid', kernel_initializer='random_uniform'))
    clf.add(Activation('elu'))
    clf.add(MaxPooling2D((2, 2)))
    clf.add(Dropout(0.3))

    clf.add(Conv2D(256, (1, 1), padding='valid', kernel_initializer='random_uniform'))
    clf.add(Activation('elu'))
    clf.add(MaxPooling2D((2, 2)))
    clf.add(Dropout(0.4))

    clf.add(Flatten())

    clf.add(Dense(1000))
    clf.add(Activation('elu'))
    clf.add(Dropout(0.5))
    clf.add(Dense(1000))
    clf.add(Activation('linear'))
    clf.add(Dropout(0.6))
    clf.add(Dense(14 * 2))
    clf.add(Reshape((14, 2)))

    clf.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return clf

def train_detector(train_gt, train_img_dir, fast_train=True):
    #print(tf.test.is_gpu_available())
    if fast_train:
        batches = 2
        epochs = 1
    else:
        batches = BATCHES
        epochs = EPOCHS
    X_train, y_train = get_train_data(train_gt, train_img_dir, fast_train)
    #print("got_train_data")
    clf = build_clf()
    clf.fit(X_train, y_train, batch_size=batches, epochs=epochs, validation_split=0.15)
    #clf.save('/content/drive/My Drive/Colab Notebooks/face_recognition/facepoints_model.hdf5')

def detect(clf, test_img_dir):
    X_test, filenames, reverse_transforms = get_test_data(test_img_dir)
    y_pred = clf.predict(X_test)
    ans = {}
    for i in range(len(y_pred)):
        y_pred[i] *= reverse_transforms[i]
        ans[filenames[i]] = np.array(y_pred[i]).flatten().tolist()
    return ans