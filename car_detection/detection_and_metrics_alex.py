# ============================== 1 Classifier model ============================


def get_cls_model(input_shape):
    """
    :param input_shape: tuple (n_rows, n_cols, n_channgels)
            input shape of image for classification
    :return: nn model for classification
    """
    # your code here \/
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

   #print(model.summary())

    return model
    # your code here /\


def fit_cls_model(X, y):
    """
    :param X: 4-dim ndarray with training images
    :param y: 2-dim ndarray with one-hot labels for training
    :return: trained nn model
    """
    # your code here \/
    model = get_cls_model((40, 100, 1))
    '''for layer in model.layers:
        print(len(layer.get_weights()))
        if len(layer.get_weights()) > 0:
            print(layer.get_weights()[1].shape)
        print(layer.input_shape)
       '''
    model.fit(X, y, epochs=10, batch_size=64)
    model.save('classifier_model.h5')

    return model
    # your code here /\


# ============================ 2 Classifier -> FCN =============================
def get_detection_model(cls_model):
    """
    :param cls_model: trained cls model
    :return: fully convolutional nn model with weights initialized from cls
             model
    """
    # your code here \/
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D
    import tensorflow as tf
    import numpy as np

    detection_model = Sequential()
    prev_flatten = False
    fl_inp_shape = None
    print(cls_model.summary())
    for i, layer in enumerate(cls_model.layers):
        if layer.name.startswith('conv2d') or layer.name.startswith('max_pooling'):
            detection_model.add(layer)
            prev_flatten = False
            fl_inp_shape = None
        elif layer.name.startswith('flatten'):
            prev_flatten = True
            fl_inp_shape = layer.input_shape
        elif prev_flatten and layer.name.startswith('dense'):
            matr, bias = layer.get_weights()
            fake, h, w, cin = fl_inp_shape
            fake, cout = layer.output_shape
            kernel_weights = matr.reshape(h, w, cin, cout)
            print(kernel_weights.shape)
            new_layer = Conv2D(cout, (h, w), activation='relu')
            if i == len(cls_model.layers) - 1:
                new_layer = Conv2D(cout, (h, w), activation='linear')
            arr = np.random.rand(h, w, cin)
            b_out = new_layer(tf.convert_to_tensor([arr]))
            new_layer.set_weights([kernel_weights, bias])
            detection_model.add(new_layer)
            prev_flatten = False
            fl_inp_shape = None
        elif not prev_flatten and layer.name.startswith('dense'):
            matr, bias = layer.get_weights()
            fake, cin = layer.input_shape
            fake, cout = layer.output_shape
            kernel_weights = matr.reshape(1, 1, cin, cout)
            new_layer = Conv2D(cout, (1, 1), activation='relu')
            if i == len(cls_model.layers) - 1:
                new_layer = Conv2D(cout, (1, 1), activation='linear')
            arr = np.random.rand(cin)
            b_out = new_layer(tf.convert_to_tensor([[[arr]]]))
            new_layer.set_weights([kernel_weights, bias])
            detection_model.add(new_layer)
            prev_flatten = False
            fl_inp_shape = None
        else:
            print('here')

    return detection_model
    # your code here /\


# ============================ 3 Simple detector ===============================
def get_detections(detection_model, dictionary_of_images):
    """
    :param detection_model: trained fully convolutional detector model
    :param dictionary_of_images: dictionary of images in format
        {filename: ndarray}
    :return: detections in format {filename: detections}. detections is a N x 5
        array, where N is number of detections. Each detection is described
        using 5 numbers: [row, col, n_rows, n_cols, confidence].
    """
    # your code here \/
    import numpy as np
    images = []
    shapes = []
    for key in dictionary_of_images:
        img = dictionary_of_images[key]
        shapes.append((img.shape[0], img.shape[1]))
        image = np.zeros((220, 370, 1))
        image[:img.shape[0], :img.shape[1], 0] = img
        images.append(image)

    images = np.array(images)
    print(detection_model.summary())
    detections = detection_model.predict(images)
    #qw = np.quantile(detections, 0.9)
    #print(np.median(detections), np.quantile(detections, 0.75), qw, np.quantile(detections, 0.99))

    result = {}
    i = 0
    for key in dictionary_of_images:
        arr = []
        imgshape0, imgshape1 = shapes[i]
        print(detections[i].shape, detections.shape)
        for j in range(detections[i].shape[0]):
            for k in range(detections[i].shape[1]):
                expsum = np.exp(detections[i][j][k][1]) + np.exp(detections[i][j][k][0])
                confidence = np.exp(detections[i][j][k][1]) / expsum
                if j * 2 + 40 <= imgshape0 and k * 2 + 100 <= imgshape1 and confidence > 0.6:
                    arr.append([j*2, k*2, 40, 100, confidence])
        result[key] = arr
        #print(sorted(arr, key=lambda h: h[4], reverse=True))
        i += 1
    return result
    # your code here /\


# =============================== 5 IoU ========================================
def calc_iou(first_bbox, second_bbox):
    """
    :param first bbox: bbox in format (row, col, n_rows, n_cols)
    :param second_bbox: bbox in format (row, col, n_rows, n_cols)
    :return: iou measure for two given bboxes
    """
    # your code here \/
    x1, y1, width1, height1 = first_bbox
    x2, y2, width2, height2 = second_bbox

    dx = min(x1 + width1, x2 + width2) - max(x1, x2)
    dy = min(y1 + height1, y2 + height2) - max(y1, y2)
    if (dx >= 0) and (dy >= 0):
        intersection = dx * dy
    else:
        intersection = 0
    union = width1 * height1 + width2 * height2 - intersection
    return intersection / union
    # your code here /\


# =============================== 6 AUC ========================================
def calc_auc(pred_bboxes, gt_bboxes):
    """
    :param pred_bboxes: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param gt_bboxes: dict of bboxes in format {filenames: bboxes}. bboxes is a
        list of tuples in format (row, col, n_rows, n_cols)
    :return: auc measure for given detections and gt
    """
    # your code here \/
    iou_thr = 0.5
    tp = []
    fp = []
    all_gt_count = 0
    for key in pred_bboxes:
        pred = pred_bboxes[key]
        gt = gt_bboxes[key]
        all_gt_count += len(gt)
        pred = sorted(pred, key=lambda h: h[4], reverse=True)
        for square in pred:
            x, y, width, height, conf = square
            max_iou = 0.0
            nummax = -1
            for i in range(len(gt)):
                iou = calc_iou((x, y, width, height), gt[i])
                if iou > max_iou:
                    max_iou = iou
                    nummax = i
            if max_iou >= iou_thr:
                del gt[nummax]
                tp.append(conf)
            else:
                fp.append(conf)
    all_arr = []
    for c in tp:
        all_arr.append((c, 'tp'))
    for c in fp:
        all_arr.append((c, 'fp'))

    all_arr = sorted(all_arr)
    tp = sorted(tp)
    result = [(0.0, 1.0, 1.0)]

    i = len(all_arr) - 1
    j = len(tp) - 1
    while i >= 0:
        c = all_arr[i][0]
        while j >= 0 and tp[j] >= c:
            j -= 1
        tpfp = (len(all_arr) - i)
        tp_count = (len(tp) - j - 1)
        precision = tp_count / tpfp
        recall = tp_count / all_gt_count
        result.append((recall, precision, c))
        i -= 1

    auc = 0
    i = len(result) - 1
    while i > 0:
        recall, precision, c = result[i]
        recall1, precision1, c1 = result[i - 1]
        auc += abs(recall1 - recall) * (precision + precision1) / 2
        i -= 1

    return auc
    # your code here /\


# =============================== 7 NMS ========================================
def nms(detections_dictionary, iou_thr=0.5):
    """
    :param detections_dictionary: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param iou_thr: IoU threshold for nearby detections
    :return: dict in same format as detections_dictionary where close detections
        are deleted
    """
    # your code here \/
    result = {}
    for key in detections_dictionary:
        arr = []
        pred = detections_dictionary[key]
        pred = sorted(pred, key=lambda h: h[4], reverse=True)
        for x, y, width, height, c in pred:
            add = True
            for x1, y1, width1, height1, c1 in arr:
                if calc_iou((x, y, width, height), (x1, y1, width1, height1)) > iou_thr:
                    add = False
            if add:
                arr.append((x, y, width, height, c))
        result[key] = arr

    return result
    # your code here /\
