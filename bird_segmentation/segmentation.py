import os
from glob import glob
import random
# from tqdm import tqdm
from datetime import datetime

random.seed(42)

import numpy as np
# import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

# from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import resize

import tensorflow as tf
# tf.executing_eagerly()
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# import albumentations as A

HEIGHT = 224
WIDTH = 224
# OUTPUT_CHANNELS = 2

BATCH_SIZE = 16
EPOCHS = 10000

IMG_PATH = '00_test_val_input/train'

exp_dir = 'seg_{}'.format(datetime.now())


# os.mkdir(exp_dir)


# import logging


class SegModel:

    def __init__(self, h, w, output_channels=1):
        self.h = h
        self.w = w
        self.output_channels = output_channels

        base_model = tf.keras.applications.MobileNetV2(weights='imagenet', input_shape=[self.h, self.w, 3],
                                                       include_top=False)

        layer_names = [
            'block_1_expand_relu',
            'block_3_expand_relu',
            'block_6_expand_relu',
            'block_13_expand_relu',
            'block_16_project',
        ]

        layers = [base_model.get_layer(name).output for name in layer_names]

        down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
        down_stack.trainable = False

        # for layer in down_stack.layers[-5:]:
        #     layer.trainable = True

        inputs = tf.keras.layers.Input(shape=[self.h, self.w, 3])
        x = inputs

        skips = down_stack(x)
        x = skips[-1]
        skips = reversed(skips[:-1])

        for up, skip in zip(self.upsample(), skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            print(x.shape, skip.shape)
            x = concat([x, skip])

        last = tf.keras.layers.Conv2DTranspose(
            self.output_channels, 3, strides=2,
            padding='same', activation='sigmoid')

        x = last(x)
        # x = tf.keras.layers.Softmax()(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=x)

    def _upsample_block(self, filters, size, norm_type='batchnorm', apply_dropout=True):

        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))

        if norm_type.lower() == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(InstanceNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.3))

        result.add(tf.keras.layers.ReLU())

        return result

    def upsample(self):
        return [
            self._upsample_block(512, 3),
            self._upsample_block(256, 3),
            self._upsample_block(128, 3),
            self._upsample_block(64, 3),
        ]


class DataGenerator(Sequence):

    def __init__(self, pair, batch_size=BATCH_SIZE, dim=(HEIGHT, WIDTH, 3), shuffle=True, transform=None):

        self.dim = dim
        self.pair = pair
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.pair) / self.batch_size))

    def __getitem__(self, index):

        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        X, y = self.__data_generation([k for k in indexes])

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.pair))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    # def form_2D_label(self, mask):
    #     out = np.zeros((mask.shape[0], mask.shape[1], 2))
    #     # print(out.shape, mask.shape)
    #     out[..., 0][mask == 0] = 1
    #     out[..., 1] = mask
    #
    #     return out

    def __data_generation(self, list_IDs_temp):

        batch_imgs = list()
        batch_labels = list()

        for i in list_IDs_temp:
            # Store sample

            # img = tf.make_ndarray(tf.image.resize(img, (HEIGHT, WIDTH)))

            # img = load_img(self.pair[i][0], target_size=self.dim)
            # img = img_to_array(img)
            # print('before img', np.unique(img))

            # label = load_img(self.pair[i][1], target_size=self.dim)

            # print('before label', np.unique(label))

            # label = label > 127

            # label = img_to_array(label)
            # label = self.form_2D_label(label)
            # print('SHAAAAPE', label.shape)
            # label = to_categorical(label, num_classes=2)
            # print('before', img.shape, label.shape)
            img = imread(self.pair[i][0])
            label = imread(self.pair[i][1])

            img = resize(img, (HEIGHT, WIDTH))
            label = resize(label, (HEIGHT, WIDTH))

            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=2)
            if len(label.shape) != 2:
                label = label[..., 0]

            if self.transform:
                transformed = self.transform(image=img, mask=label)
                img = transformed['image']
                label = transformed['mask']

            # img = img / 255.0
            # print('after', img.shape, label.shape)
            # print('after img', np.unique(img))
            # print('after label', np.unique(label))
            batch_imgs.append(img)
            batch_labels.append(label)

        return np.array(batch_imgs), np.array(batch_labels)


def get_callbacks():
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_iou',
                                                patience=5,
                                                mode='max',
                                                verbose=1,
                                                factor=0.7,
                                                min_lr=0.0000001)

    early_stop = EarlyStopping(monitor='val_iou',
                               mode='max',
                               patience=13,
                               restore_best_weights=True)

    checkpoint = ModelCheckpoint(os.path.join(exp_dir, 'segmentation_model.hdf5'),
                                 monitor='val_iou',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')

    # display = DisplayCallback()

    return [checkpoint, learning_rate_reduction, early_stop]


def get_data_split(img_dir):
    img_path = os.path.join(img_dir, 'images')
    gt_path = os.path.join(img_dir, 'gt')

    img_list = sorted(glob(os.path.join(img_path, '**/*.jpg')))
    mask_list = sorted(glob(os.path.join(gt_path, '**/*.png')))

    pairs = [[img, mask] for img, mask in zip(img_list, mask_list)]

    random.shuffle(pairs)

    # print(len(pairs))
    # for i in range(0, len(pairs), 500):
    #     print(pairs[i])

    # img_list = [os.path.basename(filename) for filename in ]
    # print(glob(os.path.join(gt_path, '**/*.png')))
    #
    # pairs = []

    train_len = int(0.8 * len(pairs))

    return pairs[:train_len], pairs[train_len:]


def iou(label, pred):
    pred = pred > 0.5
    label = label > 0.5
    intersection = tf.reduce_sum(tf.cast(pred & label, tf.float32))
    union = tf.reduce_sum(tf.cast(pred | label, tf.float32))
    return intersection / tf.maximum(union, 1e-8)


# model = SegModel(HEIGHT, WIDTH).model
# model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=[iou])
#
# train_transform = A.Compose([
#     # A.RandomCrop(widthdef get_iou=128, height=128),
#     # A.HorizontalFlip(p=0.5),
#     # A.RandomBrightnessContrast(p=0.2),
#     # # A.RandomScale(),
#     # A.ShiftScaleRotate(),
#     A.Normalize()
# ])
#
# val_transform = A.Compose([
#     A.Normalize()
# ])
#

def train_segmentation_model(img_dir):
    import os
    from glob import glob
    import random
    # from tqdm import tqdm
    from datetime import datetime

    random.seed(42)

    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow.keras.backend as K

    # from sklearn.model_selection import train_test_split
    from skimage.io import imread
    from skimage.transform import resize

    import tensorflow as tf
    # tf.executing_eagerly()
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras.utils import to_categorical, Sequence
    from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    HEIGHT = 224
    WIDTH = 224
    IMG_SHAPE = (HEIGHT, WIDTH, 3)

    def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))

        if norm_type.lower() == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(InstanceNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.3))

        result.add(tf.keras.layers.ReLU())

        return result

    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    def unet_model():
        base_model = tf.keras.applications.MobileNetV2(input_shape=[*IMG_SHAPE], include_top=False)

        # Use the activations of these layers
        layer_names = [
            'block_1_expand_relu',  # 64x64
            'block_3_expand_relu',  # 32x32
            'block_6_expand_relu',  # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',  # 4x4
        ]
        layers = [base_model.get_layer(name).output for name in layer_names]

        # Create the feature extraction model
        down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

        down_stack.trainable = False

        # for layer in down_stack.layers[-5:]:
        #     layer.trainable = True

        up_stack = [
            upsample(512, 3),  # 4x4 -> 8x8
            upsample(256, 3),  # 8x8 -> 16x16
            upsample(128, 3),  # 16x16 -> 32x32
            upsample(64, 3),  # 32x32 -> 64x64
        ]

        inputs = tf.keras.layers.Input(shape=[*IMG_SHAPE])
        x = inputs

        # Downsampling through the model
        skips = down_stack(x)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        # This is the last layer of the model
        last = tf.keras.layers.Conv2DTranspose(
            1, 3, strides=2,
            padding='same', activation='sigmoid')  # 64x64 -> 128x128

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)

    def iou(label, pred):
        pred = pred > 0.5
        label = label > 0.5
        intersection = tf.reduce_sum(tf.cast(pred & label, tf.float32))
        union = tf.reduce_sum(tf.cast(pred | label, tf.float32))
        return intersection / tf.maximum(union, 1e-8)

    def dice_coef(y_true, y_pred, smooth=1):
        """
        Dice = (2*|X & Y|)/ (|X|+ |Y|)
             =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
        ref: https://arxiv.org/pdf/1606.04797v1.pdf
        """
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)

    def dice_coef_loss(y_true, y_pred):
        return 1 - dice_coef(y_true, y_pred)

    model = unet_model()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[iou])

    train_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()

    def get_data():
        train_dir = img_dir
        train_img_path = os.path.join(train_dir, 'images')
        train_gt_path = os.path.join(train_dir, 'gt')
        train_img_list = sorted(glob(os.path.join(train_img_path, '**/*.jpg')))
        train_mask_list = sorted(glob(os.path.join(train_gt_path, '**/*.png')))

        test_dir = img_dir[:-5] + 'test'
        test_img_path = os.path.join(test_dir, 'images')
        test_gt_path = '00_test_val_gt'
        test_img_list = sorted(glob(os.path.join(test_img_path, '**/*.jpg')))
        test_mask_list = sorted(glob(os.path.join(test_gt_path, '**/*.png')))

        pairs = [[img, mask] for img, mask in zip(train_img_list, train_mask_list)]
        print('Train len: ', len(pairs))
        random.shuffle(pairs)

        X_train = []
        y_train = []

        for i, pair in enumerate(pairs):
            img = imread(pair[0])
            label = imread(pair[1])
            #
            img = resize(img, (HEIGHT, WIDTH))
            label = resize(label, (HEIGHT, WIDTH))

            # img = np.array(tf.image.resize(img, (HEIGHT, WIDTH)))
            # label = np.array(tf.image.resize(label[..., np.newaxis], (HEIGHT, WIDTH)))

            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=2)
            if len(label.shape) != 2:
                label = label[..., 0]

            X_train.append(img)
            y_train.append(label)

            if i % 100 == 0:
                print(i)

        pairs = [[img, mask] for img, mask in zip(test_img_list, test_mask_list)]
        print('Val len: ', len(pairs))
        X_val = []
        y_val = []

        for i, pair in enumerate(pairs):
            img = imread(pair[0])
            label = imread(pair[1])

            img = resize(img, (HEIGHT, WIDTH))
            label = resize(label, (HEIGHT, WIDTH))

            # img = np.array(tf.image.resize(img, (HEIGHT, WIDTH)))
            # label = np.array(tf.image.resize(label[..., np.newaxis], (HEIGHT, WIDTH)))

            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=2)
            if len(label.shape) != 2:
                label = label[..., 0]

            X_val.append(img)
            y_val.append(label)

            if i % 100 == 0:
                print(i)

        return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val)

    X_train, y_train, X_val, y_val = get_data()
    # X_train = preprocess_input(X_train)
    # X_val = preprocess_input(X_val)
    train_datagen.fit(X_train)

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_iou',
                                                mode='max',
                                                patience=7,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.000000001)
    early_stop = EarlyStopping(monitor='val_iou',
                               mode='max',
                               patience=30,
                               restore_best_weights=True)

    checkpoint = ModelCheckpoint('segmentation_model.hdf5',
                                 monitor='val_iou',
                                 mode='max',
                                 verbose=1,
                                 save_best_only=True)

    BATCH = 16
    EPOCHS = 10000

    model.fit(
        train_datagen.flow(X_train, y_train, seed=123, batch_size=BATCH),
        epochs=EPOCHS,
        validation_data=test_datagen.flow(X_val, y_val, seed=123, batch_size=BATCH),
        callbacks=[learning_rate_reduction, early_stop, checkpoint],
        workers=6)

    return model

    # print(model.summary())
    #
    # global val_pairs
    # train_pairs, val_pairs = get_data_split(img_dir)
    # print('00_test_val_input/train/gt/045.Northern_Fulmar/Northern_Fulmar_0063_43631.png' in [j[1] for j in val_pairs])
    # # print(val_pairs)
    # # exit()
    # # print('LEEEEEEEEEEEEN', len(train_pairs), len(val_pairs))
    #
    # train_dataset = DataGenerator(train_pairs, shuffle=True)#, transform=train_transform)
    # global val_dataset
    # val_dataset = DataGenerator(val_pairs, shuffle=False)#, transform=val_transform)
    #
    # callbacks = get_callbacks()
    #
    # model.fit(train_dataset,
    #           validation_data=val_dataset,
    #           epochs=EPOCHS,
    #           callbacks=callbacks)
    #
    # return model
    # return 1


# from skimage.transform import rotate


def shift_image(X, dx, dy):
    X = np.roll(X, dy, axis=0)
    X = np.roll(X, dx, axis=1)
    if dy > 0:
        X[:dy, :] = 0
    elif dy < 0:
        X[dy:, :] = 0
    if dx > 0:
        X[:, :dx] = 0
    elif dx < 0:
        X[:, dx:] = 0
    return X


def predict(model, filename):
    img = imread(filename)
    orig_size = img.shape
    # print(orig_size)

    img = resize(img, (HEIGHT, WIDTH))
    # print(img.shape)
    #        label = resize(label, (HEIGHT, WIDTH))

    # if len(img.shape) == 2:
    #             img = np.stack([img, img, img], axis=2)imga = load_img(filename)
    # print(imga.shape)

    # imga = tf.image.resize(imga, (HEIGHT, WIDTH))

    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=2)

    # img = val_transform(imga)

    # img = img_to_array(imga) / 255.

    #
    # mask0 = model(np.clip(img[np.newaxis, ...] + 0.015, 0, 1))[0]
    mask1 = model(img[np.newaxis, ...])[0]
    mask2 = model(img[:, ::-1][np.newaxis, ...])[0][:, ::-1]
    mask3 = shift_image(model(shift_image(img, 7, -7)[np.newaxis, ...])[0], -7, 7)
    mask4 = shift_image(model(shift_image(img, -5, 6)[np.newaxis, ...])[0], 5, -6)
    # mask5 = shift_image(model(shift_image(img, -7, 7)[np.newaxis, ...])[0], 7, -7)
    # mask6 = shift_image(model(shift_image(img, 10, -2)[np.newaxis, ...])[0], -10, 2)
    # img = np.array(img)
    # mask3 = rotate(np.array(model(rotate(img, 5)[np.newaxis, ...])[0]), -5)
    # mask4 = rotate(np.array(model(rotate(img, -5)[np.newaxis, ...])[0]), 5)

    # batch = np.array([img for i in range(8)])

    # res = np.zeros(img.shape)
    # for i in range(5):
    # test_datagen = ImageDataGenerator(horizontal_flip=True)
    # res = model.predict_generator(test_datagen.flow(batch)).mean(axis=0)

    # res /= 5

    # res = tf.image.resize(res, (orig_size[0], orig_size[1]))
    #
    # return np.array(res) + 0.1

    mask = tf.image.resize((mask1 + mask2 + mask3 + mask4) / 4, (orig_size[0], orig_size[1]))
    # print(mask.shape)
    return np.array(mask) + 0.1

    # raw_iou = get_iou(label, res[0])
    # postp = postprocess_mask(res[0])
    # post_iou = get_iou(label, postp)
    # display([imga, label, res[0], postp], raw_iou, post_iou)
    # pass


# from contextlib import redirect_stdout

GPU = 0
if __name__ == "__main__":
    # f = io.StringIO()
    # with redirect_stdout(f):
    with tf.device(f'/device:gpu:{GPU}'):
        train_segmentation_model(IMG_PATH)

# @tf.function
# def iou_metrics(y_true, y_pred):
#     iou.reset_states()
#     iou.update_state(create_mask(y_true), create_mask(y_pred))
#
#     print(iou.result())
#
#     return iou.result().numpy()
# def DiceBCELoss(targets, inputs, smooth=1e-6):
#     print(targets.shape, inputs.shape)
#     # flatten label and prediction tensors
#     inputs = K.flatten(inputs)
#     targets = K.flatten(targets)
#
#     BCE = K.binary_crossentropy(targets, inputs)
#     print(targets.shape, inputs.shape)
#     intersection = K.sum(inputs * targets)
#     dice_loss = 1 - (2 * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
#     Dice_BCE = BCE + dice_loss
#
#     return Dice_BCE

# def create_mask(pred_mask):
#     pred_mask = tf.argmax(pred_mask, axis=-1)
#     pred_mask = pred_mask[..., tf.newaxis]
#     return pred_mask[0]


# iou = tf.keras.metrics.MeanIoU(1, name=None, dtype=None)

# def get_iou(gt, pred):
#     # print(gt)
#     # print(pred)
#     # with sess.as_default():
#     # pred = (pred > 0.5)#.astype('uint8')
#     # gt = tf.cast(gt, tf.bool)
#     #
#     # a = (gt * pred).sum()
#     # b = tf.clip_by_value(gt + pred, 0, 1).sum()
#
#     pred = tf.keras.backend.round(pred)
#
#     return tf.keras.backend.sum(tf.bitwise.bitwise_and(gt, pred)) / tf.keras.backend.sum(
#         tf.bitwise.bitwise_or(gt, pred))

# return tf.keras.backend.sum(gt * pred) / tf.keras.backend.sum(tf.keras.backend.clip(gt + pred, 0, 1))


# def show_predictions(dataset=None, num=1):
#     if dataset:
#         for image, mask in dataset.take(num):
#             pred_mask = model.predict(image)
#             display([image[0], mask[0], create_mask(pred_mask)])
#     else:
#         display([sample_image, sample_mask,
#                  create_mask(model.predict(sample_image[tf.newaxis, ...]))])
#
#


# def create_logger(exp_dir):
#     logger = logging.getLogger()
#     handler = logging.FileHandler(os.path.join(exp_dir, 'out.log'))
#     formatter = logging.Formatter('%(message)s')
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
#     logger.addHandler(logging.StreamHandler())
#     logger.setLevel(logging.INFO)
#
#
# create_logger(exp_dir)


# def display(display_list):
#     plt.figure(figsize=(15, 15))
#
#     title = ['Input Image', 'True Mask', 'Predicted Mask']
#
#     for i in range(len(display_list)):
#         plt.subplot(1, len(display_list), i + 1)
#         plt.title(title[i])
#         plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
#         plt.axis('off')
#     plt.show()


# def create_mask(pred_mask):
#     # pred_mask = tf.argmax(pred_mask, axis=-1)
#     pred_mask = pred_mask[...]
#     return pred_mask


# def display(display_list, raw_iou, post_iou):
#     plt.figure(figsize=(15, 15))
#     title = ['Input Image', 'True Mask', f'Predicted Mask - {raw_iou}', f'Postprocessed Mask - {post_iou}']
#
#     for i in range(len(display_list)):
#         # print(np.array(display_list[i]).shape, np.array(display_list[i]).max(), np.array(display_list[i]).min())
#         plt.subplot(1, len(display_list), i + 1)
#         plt.title(title[i])
#         # if i < 1:
#         #     a = tf.keras.preprocessing.image.array_to_img(display_list[i])
#         # else:
#         #     a = display_list[i]
#         # print(np.array(a).shape, np.array(a).min(), np.array(a).max())
#         plt.imshow(display_list[i])
#         plt.axis('off')
#
#     plt.savefig(os.path.join(exp_dir, 'viz_{}.png'.format(datetime.now())))
#     plt.close('all')
#
#
# def postprocess_mask(mask):
#     return mask > 0.01


# def show_predictions():
#     ious = 0
#     t = 0
#     for i in range(len(val_dataset)):
#         # print(i)
#         X, y = val_dataset[i]
#
#         pred = model.predict(X)
#
#         for y_sample, pred_sample in zip(y, pred):
#             # print(type(y_sample), type(pred_sample))
#             # y_sample = tf.keras.backend.eval(y_sample)
#             # pred_sample = tf.keras.backend.eval(pred_sample)
#             y_sample = np.uint64(y_sample)
#             pred_sample = np.uint64(pred_sample)[..., 0]
#             # print(type(y_sample), type(pred_sample))
#             # print(y_sample.shape, y_sample.min(), y_sample.max())
#             # print(pred_sample.shape, pred_sample.min(), pred_sample.max())
#             shiou = iou(y_sample, pred_sample)
#             # print(shiou)
#             ious += shiou
#             t += 1
#
#     ious /= t
#
#     print('SHAH IOU', ious)
#     print('00_test_val_input/train/gt/045.Northern_Fulmar/Northern_Fulmar_0063_43631.png' in [j[1] for j in val_pairs])
#
#     for i in [1, 100, 1000]:
#
#         imga = load_img(val_pairs[i][0],
#                         target_size=(HEIGHT, WIDTH))
#         # mask = load_img('00_test_val_input/train/gt/045.Northern_Fulmar/Northern_Fulmar_0063_43631.png',
#         #                 target_size=(128, 128))
#
#         label = imread(val_pairs[i][1])
#         if len(label.shape) > 2:
#             label = label[..., 0]
#         # label = label > 127
#         label = resize(label, (HEIGHT, WIDTH))
#
#         img = img_to_array(imga) / 255.
#         if len(img.shape) == 2:
#             img = np.stack([img, img, img], axis=2)
#
#         res = model(img[np.newaxis, ...])
#         # print('SHAAAAAAAAAAAAAAAAAAAAAPE', res.shape)
#         r = np.array(res[0])
#         if len(r.shape) > 2:
#             r = r[..., 0]
#         raw_iou = iou(label, r)
#         postp = postprocess_mask(r)
#         post_iou = iou(label, postp)
#         display([imga, label, r, postp], raw_iou, post_iou)
#
#     # print(res.shape)
#
#
# class DisplayCallback(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         show_predictions()
#         print('\nSample Prediction after epoch {}\n'.format(epoch + 1))
#
# def on_epoch_start(self, epoch, logs=None):
#     show_predictions()
#     print('\nSample Prediction after epoch {}\n'.format(epoch + 1))


# class InstanceNormalization(tf.keras.layers.Layer):
#
#     def __init__(self, epsilon=1e-5):
#         super(InstanceNormalization, self).__init__()
#         self.epsilon = epsilon
#
#     def build(self, input_shape):
#         self.scale = self.add_weight(
#             name='scale',
#             shape=input_shape[-1:],
#             initializer=tf.random_normal_initializer(1., 0.02),
#             trainable=True)
#
#         self.offset = self.add_weight(
#             name='offset',
#             shape=input_shape[-1:],
#             initializer='zeros',
#             trainable=True)
#
#     def call(self, x):
#         mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
#         inv = tf.math.rsqrt(variance + self.epsilon)
#         normalized = (x - mean) * inv
#         return self.scale * normalized + self.offset


# def get_iou(true, pred):
#     true = tf.keras.backend.cast(true > 0.5, dtype='float32')
#     pred = tf.keras.backend.cast(pred > 0.5, dtype='float32')
#
#     a = K.stack([true, pred], axis=0)
#     intersection = K.cast(K.all(a, axis=0), dtype='float32')
#     union = K.cast(K.any(a, axis=0), dtype='float32')
#
#     # intersection = true * pred
#     #
#     #
#     # union
#     #
#     # notTrue = 1 - true
#     # union = true + (notTrue * pred)
#
#     return K.sum(intersection) / K.sum(union)


# def shah_iou(gt, pred):
#     # print(type(gt))
#     # print(gt.shape)
#     # print(type(pred))
#     # print(pred.shape)
#     gt = np.array(gt, dtype=bool)
#     pred = np.array(pred, dtype=bool)
#     # pred = np.bool(pred)
#     a = (gt & pred).sum()
#     b = (gt | pred).sum()
#     # print(a, b)
#     return a / b


# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=[iou, 'val_loss'])


# model.compile(optimizer='adam', loss=DiceBCELoss, metrics=[get_iou])
