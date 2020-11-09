# ============================== 1 Classifier model ============================

def get_cls_model(input_shape):
    """
    :param input_shape: tuple (n_rows, n_cols, n_channgels)
            input shape of image for classification
    :return: nn model for classification
    """
    # your code here \/
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Softmax, Flatten, InputLayer, Conv2D, MaxPool2D
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
    # your code here /\

def fit_cls_model(X, y):
    """
    :param X: 4-dim ndarray with training images
    :param y: 2-dim ndarray with one-hot labels for training
    :return: trained nn model
    """
    model = get_cls_model((40, 100, 1))
    batches = 128
    epochs = 8
    model.fit(X, y, batch_size=batches, epochs=epochs)
    #model.save('classifier_model.h5')
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
    from tensorflow.keras.layers import Conv2D, Dense, Flatten, InputLayer
    from tensorflow import convert_to_tensor
    import numpy as np

    model = Sequential()
    ind = 0
    model.add(InputLayer(input_shape=(220, 370, 1)))
    while True:
        try:
            cur = cls_model.get_layer(index=ind)
        except ValueError:
            break
        ind += 1
        if isinstance(cur, Dense):
            weights, biases = cur.get_weights()
            _, cin = cur.input_shape
            cout = cur.units
            if cur.activation == 'softmax':
                act = 'linear'
            else:
                act = cur.activation
            layer = Conv2D(cout, (1, 1), activation=act)
            layer(convert_to_tensor([[[np.random.rand(cin)]]]))
            layer.set_weights([weights.reshape(1, 1, cin, cout), biases])
            model.add(layer)
        elif isinstance(cur, Flatten):
            dense = cls_model.get_layer(index=ind)
            ind += 1
            weights, biases = dense.get_weights()
            _, h, w, cin = cur.input_shape
            cout = dense.units
            if dense.activation == 'softmax':
                act = 'linear'
            else:
                act = dense.activation
            layer = Conv2D(cout, (h, w), activation=act)
            layer(convert_to_tensor([np.random.rand(h, w, cin)]))
            layer.set_weights([weights.reshape(h, w, cin, cout), biases])
            model.add(layer)
        elif not isinstance(cur, InputLayer):
            model.add(cur)
    return model
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
    import numpy as np
    from skimage.transform import resize
    # your code here \/
    images = np.zeros((len(dictionary_of_images), 220, 370, 1))
    prev_shapes = []
    i = 0
    for filename in dictionary_of_images:
        img = dictionary_of_images[filename]
        prev_shapes.append(img.shape)
        images[i, :img.shape[0]:, :img.shape[1]:, 0] = img
        i += 1
    pred = detection_model.predict(images)
    detections = {}
    i = 0
    for filename in dictionary_of_images:
        detection = []
        for j in range(pred[i].shape[0]):
            for k in range(pred[i].shape[1]):
                probability = np.exp(np.exp(pred[i][j][k][1])) / (np.exp(pred[i][j][k][1]) + np.exp(pred[i][j][k][0]))
                if probability > 0.7 and 2 * j + 40 <= prev_shapes[i][0] and 2 * k + 100 <= prev_shapes[i][1]:
                    detection.append([2 * j, 2 * k, 40, 100, probability])
        detections[filename] = detection
        i += 1
    return detections
    # your code here /\


# =============================== 5 IoU ========================================
def calc_iou(first_bbox, second_bbox):
    """
    :param first bbox: bbox in format (row, col, n_rows, n_cols)
    :param second_bbox: bbox in format (row, col, n_rows, n_cols)
    :return: iou measure for two given bboxes
    """
    # your code here \/
    r1, c1, h1, w1 = first_bbox
    r2, c2, h2, w2 = second_bbox
    height = min(r1 + h1, r2 + h2) - max(r1, r2)
    width = min(c1 + w1, c2 + w2) - max(c1, c2)
    inter = height * width if height > 0 and width > 0 else 0
    union = h1 * w1 + h2 * w2 - inter
    return inter / union
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
    import numpy as np
    tp = []
    fp = []
    sum_lens = 0
    iou_thr = 0.5
    for filename in pred_bboxes:
        gt = gt_bboxes[filename]
        pred = sorted(pred_bboxes[filename], key=lambda x: x[4], reverse=True)
        sum_lens += len(gt)
        for detection in pred:
            r, c, h, w, p = detection
            gt = sorted(gt, key=lambda x: calc_iou((r, c, h, w), x))
            if len(gt) == 0:
                break
            elif len(gt) == 1:
                ind = 0
            else:
                ind = -1
            if calc_iou((r, c, h, w), gt[ind]) >= iou_thr:
                tp.append(p)
                gt.pop()
            else:
                fp.append(p)
    all_prob = np.array(np.concatenate((tp, fp)), dtype='float')
    all_prob.sort()
    tp = np.array(tp)
    tp.sort()
    pnts = []
    for probability in all_prob:
        count_tp = np.sum(tp >= probability)
        count_all = np.sum(all_prob >= probability)
        recall = count_tp / sum_lens
        precision = count_tp / count_all
        pnts.append((recall, precision, probability))
    pnts.append((0.0, 1.0, 1.0))
    auc = 0
    for i in range(len(pnts) - 1):
        r1, p1, prob1 = pnts[i]
        r2, p2, prob2 = pnts[i + 1]
        auc += abs(r1 - r2) * (p1 + p2) / 2
    return auc
    # your code here /\


# =============================== 7 NMS ========================================
def nms(detections_dictionary, iou_thr=0.6):
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
    import numpy as np
    ans = {}
    for filename in detections_dictionary:
        detections = sorted(detections_dictionary[filename], key=lambda x: x[4], reverse=True)
        mask = np.ones(len(detections))
        for i in range(len(detections)):
            if mask[i]:
                for j in range(i+1, len(detections), 1):
                    r1, c1, h1, w1, p1 = detections[i]
                    r2, c2, h2, w2, p2 = detections[j]
                    if calc_iou((r1, c1, h1, w1), (r2, c2, h2, w2)) > iou_thr:
                        mask[j] = 0
        new_detections = []
        for i in range(len(detections)):
            if mask[i]:
                new_detections.append(detections[i])
        ans[filename] = new_detections
    return ans
    # your code here /\
