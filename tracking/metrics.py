
def iou_score(first_bbox, second_bbox):
    """Jaccard index or Intersection over Union.

    https://en.wikipedia.org/wiki/Jaccard_index

    bbox: [xmin, ymin, xmax, ymax]
    """

    assert len(first_bbox) == 4
    assert len(second_bbox) == 4

    r1, c1, h1, w1 = first_bbox
    h1 -= r1
    w1 -= c1
    r2, c2, h2, w2 = second_bbox
    h2 -= r2
    w2 -= c2
    height = min(r1 + h1, r2 + h2) - max(r1, r2)
    width = min(c1 + w1, c2 + w2) - max(c1, c2)
    inter = height * width if height > 0 and width > 0 else 0
    union = h1 * w1 + h2 * w2 - inter
    return inter / union


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
        obj_dict = {obj[0]: obj[1:] for obj in frame_obj}
        hyp_dict = {hyp[0]: hyp[1:] for hyp in frame_hyp}
        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        for id in matches.keys():
            if (id in obj_dict.keys()) and (matches[id] in hyp_dict.keys()) and (iou_score(obj_dict[id], hyp_dict[matches[id]]) > threshold):
                dist_sum += iou_score(obj_dict[id], hyp_dict[matches[id]])
                match_count += 1
                del obj_dict[id]
                del hyp_dict[matches[id]]
        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        pairwise = []
        for ido in obj_dict.keys():
            for idh in hyp_dict.keys():
                if iou_score(obj_dict[ido], hyp_dict[idh]) > threshold:
                    pairwise.append([ido, idh, iou_score(obj_dict[ido], hyp_dict[idh])])
        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        pairwise.sort(key=lambda x: -x[2])
        for pair in pairwise:
            if (not pair[0] in matches.keys()) and (not pair[1] in matches.values()):
                dist_sum += pair[2]
                match_count += 1
                matches[pair[0]] = pair[1]
        # Step 5: Update matches with current matched IDs

    # Step 6: Calculate MOTP
    MOTP = dist_sum / match_count

    return MOTP


def motp_mota(obj, hyp, threshold=0.3):
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
    obj_sum = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Step 1: Convert frame detections to dict with IDs as keys
        obj_sum += len(frame_obj)
        obj_dict = {obj[0]: obj[1:] for obj in frame_obj}
        hyp_dict = {hyp[0]: hyp[1:] for hyp in frame_hyp}
        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        for id in matches.keys():
            if (id in obj_dict.keys()) and (matches[id] in hyp_dict.keys()) and (iou_score(obj_dict[id], hyp_dict[matches[id]]) > threshold):
                dist_sum += iou_score(obj_dict[id], hyp_dict[matches[id]])
                match_count += 1
                del obj_dict[id]
                del hyp_dict[matches[id]]
        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        pairwise = []
        for ido in obj_dict.keys():
            for idh in hyp_dict.keys():
                if iou_score(obj_dict[ido], hyp_dict[idh]) > threshold:
                    pairwise.append([ido, idh, iou_score(obj_dict[ido], hyp_dict[idh])])
        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        pairwise.sort(key=lambda x: -x[2])
        new_matches = []
        for pair in pairwise:
            dist_sum += pair[2]
            match_count += 1
            del obj_dict[pair[0]]
            del hyp_dict[pair[1]]
            new_matches.append(pair[:2])           
        # Step 5: If matched IDs contradict previous matched IDs - increase mismatch error
        inv_matches = {v: k for k, v in matches.items()}
        for match in new_matches:
            if (match[0] in matches.keys() and matches[match[0]] != match[1]) or (match[1] in matches.values() and inv_matches[match[1]] != match[0]):
                mismatch_error += 1
            matches[match[0]] = match[1]
        # Step 6: Update matches with current matched IDs
        # Step 7: Errors
        # All remaining hypotheses are considered false positives
        # All remaining objects are considered misses
        false_positive += len(hyp_dict)
        missed_count += len(obj_dict)

    # Step 8: Calculate MOTP and MOTA
    MOTP = dist_sum / match_count
    MOTA = 1 - (mismatch_error + missed_count + false_positive) / obj_sum

    return MOTP, MOTA
