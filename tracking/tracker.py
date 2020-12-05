import numpy as np
import os

from moviepy.editor import VideoFileClip

from detection import detection_cast, extract_detections, draw_detections
from metrics import iou_score


class Tracker:
    """Generate detections and build tracklets."""
    def __init__(self, return_images=True, lookup_tail_size=80, labels=None):
        self.return_images = return_images  # Return images or detections?
        self.frame_index = 0
        self.labels = labels  # Tracker label list
        self.detection_history = []  # Saved detection list
        self.last_detected = {}
        self.tracklet_count = 0  # Counter to enumerate tracklets

        # We will search tracklet at last lookup_tail_size frames
        self.lookup_tail_size = lookup_tail_size

    def new_label(self):
        """Get new unique label."""
        self.tracklet_count += 1
        return self.tracklet_count - 1

    def init_tracklet(self, frame):
        """Get new unique label for every detection at frame and return it."""
        # Write code here
        # Use extract_detections and new_label
        detections = extract_detections(frame)
        for detection in detections:
            detection[0] = self.new_label()
        return detections

    @property
    def prev_detections(self):
        """Get detections at last lookup_tail_size frames from detection_history.

        One detection at one id.
        """
        ids = []
        detections = []
        for i in range(min(self.lookup_tail_size, len(self.detection_history))):
            for detection in self.detection_history[-i-1]:
                if not detection[0] in ids:
                    ids.append(detection[0])
                    detections.append(detection)
        return detection_cast(detections)

    def bind_tracklet(self, detections):
        """Set id at first detection column.

        Find best fit between detections and previous detections.

        detections: numpy int array Cx5 [[label_id, xmin, ymin, xmax, ymax]]
        return: binded detections numpy int array Cx5 [[tracklet_id, xmin, ymin, xmax, ymax]]
        """
        detections = detections.copy()
        prev_detections = self.prev_detections

        # Write code here

        # Step 1: calc pairwise detection IOU
        pairwise = []
        used_ids = []
        for i, cur_detection in enumerate(detections):
            for j, prev_detection in enumerate(prev_detections):
                pairwise.append([i, j, iou_score(cur_detection[1:], prev_detection[1:])])
        # Step 2: sort IOU list
        pairwise.sort(key=lambda x: -x[2])
        # Step 3: fill detections[:, 0] with best match
        # One matching for each id
        for pair in pairwise:
            curr_id = prev_detections[pair[1]][0]
            if not curr_id in used_ids:
                used_ids.append(curr_id)
                detections[pair[0]][0] = curr_id
        # Step 4: assign new tracklet id to unmatched detections
        for detection in detections:
            if detection[0] == -1:
                detection[0] = self.new_label()
        return detection_cast(detections)

    def save_detections(self, detections):
        """Save last detection frame number for each label."""
        for label in detections[:, 0]:
            self.last_detected[label] = self.frame_index

    def update_frame(self, frame):
        if not self.frame_index:
            # First frame should be processed with init_tracklet function
            detections = self.init_tracklet(frame)
        else:
            # Every Nth frame should be processed with CNN (very slow)
            # First, we extract detections
            detections = extract_detections(frame, labels=self.labels)
            # Then bind them with previous frames
            # Replacing label id to tracker id is performing in bind_tracklet function
            detections = self.bind_tracklet(detections)

        # After call CNN we save frame number for each detection
        self.save_detections(detections)
        # Save detections and frame to the history, increase frame counter
        self.detection_history.append(detections)
        self.frame_index += 1

        # Return image or raw detections
        # Image usefull to visualizing, raw detections to metric
        if self.return_images:
            return draw_detections(frame, detections)
        else:
            return detections


def main():
    dirname = os.path.dirname(__file__)
    input_clip = VideoFileClip(os.path.join(dirname, 'data', 'test.mp4'))

    tracker = Tracker()
    input_clip.fl_image(tracker.update_frame).preview()


if __name__ == '__main__':
    main()
