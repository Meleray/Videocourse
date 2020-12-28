import numpy as np

def iou_score(bbox1, bbox2):
    """Jaccard index or Intersection over Union.

    https://en.wikipedia.org/wiki/Jaccard_index

    bbox: [xmin, ymin, xmax, ymax]
    """

    assert len(bbox1) == 4
    assert len(bbox2) == 4

    # Write code here
    r = max(0, min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]))
    c = max(0, min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]))
    ins = r * c
    return ins / (((bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])) + ((bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])) - ins)


def motp(obj, hyp, threshold=0.5):
    """Calculate MOTP

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Write code here

        # Step 1: Convert frame detections to dict with IDs as keys
        do = {frame_obj[i][0]: frame_obj[i][1:] for i in range(len(frame_obj))}
        dh = {frame_hyp[i][0]: frame_hyp[i][1:] for i in range(len(frame_hyp))}
        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        for key in do.copy().keys():
            if (dh.get(key)):
                IoU = iou_score(do[key], dh[key])
                if (IoU > threshold):
                    dist_sum += IoU
                    match_count += 1
                    do.pop(key)
                    dh.pop(key)
        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        list_iou = []
        for keyo in do.keys():
            for keyh in dh.keys():
                IoU = iou_score(do[keyo], dh[keyh])
                if (IoU > threshold):
                    list_iou.append([keyo, keyh, IoU])
        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        list_iou = np.array(list_iou)
        if (len(list_iou) > 0):
            list_iou = list_iou[list_iou[:, 2].argsort()[::-1]]
        for i in range(len(list_iou)):
            dist_sum += list_iou[i][2]
            match_count += 1
            do.pop(list_iou[i][0])
            dh.pop(list_iou[i][1])
        # Step 5: Update matches with current matched IDs

        pass

    # Step 6: Calculate MOTP
    MOTP = dist_sum / match_count

    return MOTP


def motp_mota(obj, hyp, threshold=0.4):
    """Calculate MOTP/MOTA

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0
    missed_count = 0
    false_positive = 0
    mismatch_error = 0

    matches = {}  # matches between object IDs and hypothesis IDs
    
    gt = 0

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        gt += len(frame_obj)
        # Step 1: Convert frame detections to dict with IDs as keys
        do = {b[0]: b[1:] for b in frame_obj}
        dh = {b[0]: b[1:] for b in frame_hyp}

        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        for ido, idh in matches.items():
            if ((ido in do) and (idh in dh)):
                IoU = iou_score(do[ido], dh[idh])
                if (IoU > threshold):
                    dist_sum += IoU
                    match_count += 1
                    do.pop(ido)
                    dh.pop(idh)
        '''
        for key in do.copy().keys():
            if (dh.get(key)):
                IoU = iou_score(do[key], dh[key])
                if (IoU > threshold):
                    dist_sum += IoU
                    match_count += 1
                    do.pop(key)
                    dh.pop(key)
        '''
        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        list_iou = []
        for keyo in do.keys():
            for keyh in dh.keys():
                IoU = iou_score(do[keyo], dh[keyh])
                if (IoU > threshold):
                    list_iou.append([keyo, keyh, IoU])
                    break
        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        list_iou = np.array(list_iou)
        if (len(list_iou) > 0):
            list_iou = list_iou[list_iou[:, 2].argsort()[::-1]]
        for i in range(len(list_iou)):
            ido = list_iou[i][0]
            idh = list_iou[i][1]
            if ((ido in matches) and (matches[ido] != idh)):
                mismatch_error += 1
            matches.update({ido: idh})
            dist_sum += list_iou[i][2]
            match_count += 1
            do.pop(ido)
            dh.pop(idh)

        # Step 5: If matched IDs contradict previous matched IDs - increase mismatch error

        # Step 6: Update matches with current matched IDs

        # Step 7: Errors
        # All remaining hypotheses are considered false positives
        false_positive += len(dh)
        # All remaining objects are considered misses
        missed_count += len(do)
        pass

    # Step 8: Calculate MOTP and MOTA
    MOTP = dist_sum / match_count
    MOTA = 1 - (missed_count + false_positive + mismatch_error) / gt

    return MOTP, MOTA