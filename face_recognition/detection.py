import os
import numpy as np
from skimage.io import imread
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D,MaxPool2D, ZeroPadding2D
from skimage.transform import resize, rotate
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split


IMG_SIZE = 100
EPOCHS = 80
BATCHES = 128

def get_data(data_dir, index, size=(IMG_SIZE, IMG_SIZE, 1), fast_train=True):
    gt = pd.DataFrame(index).transpose()
    files = sorted(os.listdir(data_dir))
    images = []
    pnts = []
    if fast_train:
        n_files = 3
    else:
        n_files = len(files)
    for i in range(n_files):
        img = imread(data_dir+'/'+files[i])
        x, y = img.shape[0], img.shape[1]
        img = resize(rgb2gray(img), size, mode='constant')
        cur_pnts = gt.iloc[i, :].values
        xs = cur_pnts[::2].astype('float')
        ys = cur_pnts[1::2].astype('float')
        xs = np.rint(xs / x * size[0])
        ys = np.rint(ys / y * size[1])
        cur_pnts[::2] = xs
        cur_pnts[1::2] = ys
        pnts.append(cur_pnts)
        images.append(img)
    images = np.array(images, dtype='float')
    pnts = np.array(pnts)
    return images, pnts

def flip_img(img, cur_pnts):
    new_img = np.zeros((IMG_SIZE, IMG_SIZE, 1))
    new_img[:, :, 0] = np.fliplr(img[:, :, 0])
    xs = IMG_SIZE - cur_pnts[::2]
    ys = cur_pnts[1::2]
    new_pnts = np.zeros(np.shape(cur_pnts))
    new_pnts[0] = xs[3]
    new_pnts[1] = ys[3]
    new_pnts[2] = xs[2]
    new_pnts[3] = ys[2]
    new_pnts[4] = xs[1]
    new_pnts[5] = ys[1]
    new_pnts[6] = xs[0]
    new_pnts[7] = ys[0]
    new_pnts[8] = xs[9]
    new_pnts[9] = ys[9]
    new_pnts[10] = xs[8]
    new_pnts[11] = ys[8]
    new_pnts[12] = xs[7]
    new_pnts[13] = ys[7]
    new_pnts[14] = xs[6]
    new_pnts[15] = ys[6]
    new_pnts[16] = xs[5]
    new_pnts[17] = ys[5]
    new_pnts[18] = xs[4]
    new_pnts[19] = ys[4]
    new_pnts[20] = xs[10]
    new_pnts[21] = ys[10]
    new_pnts[22] = xs[13]
    new_pnts[23] = ys[13]
    new_pnts[24] = xs[12]
    new_pnts[25] = ys[12]
    new_pnts[26] = xs[11]
    new_pnts[27] = ys[11]
    return new_img, new_pnts

def rotate_img(image, pnts, angle):
    phi = angle * np.pi / 180
    rotation_matrix = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
    new_pnts = np.array(pnts.reshape(14, 2))
    img = rotate(image, angle, center=(0,0), resize=False, mode='constant', order=1)
    for i in range(14):
      new_pnts[i] = np.dot(rotation_matrix, new_pnts[i].T)
    return img, new_pnts.flatten()

def data_augmentation(X, y):
    X_new, y_new = np.zeros(np.shape(X)), np.zeros(np.shape(y))
    n_imgs = X.shape[0]
    for i in range(n_imgs):
        X_tmp, y_tmp = flip_img(X[i], y[i])
        X_new[i] = X_tmp
        y_new[i] = y_tmp
    X_aug = np.vstack((X, X_new))
    y_aug = np.vstack((y, y_new))
    n_imgs = X_aug.shape[0]
    X_new, y_new = np.zeros((n_imgs, 100, 100, 1)), np.zeros((n_imgs, 28))
    for i in range(n_imgs):
      X_tmp, y_tmp = rotate_img(X_aug[i], y_aug[i], -15)
      X_new[i] = X_tmp
      y_new[i] = y_tmp
    X_aug = np.vstack((X_aug, X_new))
    y_aug = np.vstack((y_aug, y_new))
    print(X_aug.shape, y_aug.shape)
    return X_aug, y_aug

def get_model():
    model = Sequential()
    model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=(100,100,1)))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(28))
    return model


def train_detector(train_gt, train_img_dir, fast_train=True):
    X, y = get_data(train_img_dir, train_gt, fast_train=fast_train)
    model = get_model()
    model.compile(optimizer = 'adam', loss = 'mse')
    reductor = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 5, verbose = 1)
    if fast_train:
        batches = 2
        epochs = 1
        model.fit(X.astype('float'), y.astype('float'), batch_size=batches, shuffle=True, epochs=epochs, callbacks=[reductor])
    else:
        batches = BATCHES
        epochs = EPOCHS
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
        X_train, y_train = data_augmentation(X_train, y_train)
        X_val, y_val = data_augmentation(X_val, y_val)
        X_train = (X_train - np.mean(X_train)) / np.std(X_train)
        X_val = (X_val - np.mean(X_val)) / np.std(X_val)
        model.fit(X_train.astype('float'), y_train.astype('float'), batch_size=batches, shuffle=True, epochs=epochs, validation_data=(X_val.astype('float'), y_val.astype('float')), callbacks=[reductor])
    #model.save('facepoints_model.hdf5')

def get_test_data(data_dir, size=(IMG_SIZE, IMG_SIZE, 1)):
    files = sorted(os.listdir(data_dir))
    images = []
    for i in range(len(files)):
        img = imread(data_dir+'/'+files[i])
        img = resize(rgb2gray(img), size, mode='constant')
        images.append(img)
    images = np.array(images, dtype='float')
    return images, files

def detect(clf, test_img_dir):
    X_test, filenames = get_test_data(test_img_dir)
    y_pred = clf.predict(X_test)
    ans = {}
    for i in range(len(y_pred)):
        ans[filenames[i]] = np.array(y_pred[i]).flatten().tolist()
    return ans