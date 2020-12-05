import numpy as np
import os

from moviepy.editor import VideoFileClip
from skimage.feature import match_template
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

from detection import extract_detections, draw_detections, detection_cast
from tracker import Tracker


def gaussian(shape, x, y, dx, dy):
    """Return gaussian for tracking.

    shape: [width, height]
    x, y: gaussian center
    dx, dy: std by x and y axes

    return: numpy array (width x height) with gauss function, center (x, y) and std (dx, dy)
    """
    Y, X = np.mgrid[0:shape[0], 0:shape[1]]
    return np.exp(-(X - x) ** 2 / dx ** 2 - (Y - y) ** 2 / dy ** 2)


class CorrelationTracker(Tracker):
    """Generate detections and building tracklets."""
    def __init__(self, detection_rate=5, **kwargs):
        super().__init__(**kwargs)
        self.detection_rate = detection_rate  # Detection rate
        self.prev_frame = None  # Previous frame (used in cross correlation algorithm)

    def build_tracklet(self, frame):
        """Between CNN execution uses normalized cross-correlation algorithm (match_template)."""
        detections = []
        # Write code here
        # Apply rgb2gray to frame and previous frame
        prev_frame = rgb2gray(self.prev_frame)
        frame = rgb2gray(frame)
        # For every previous detection
        # Use match_template + gaussian to extract detection on current frame
        for label, xmin, ymin, xmax, ymax in self.detection_history[-1]:
            # Step 0: Extract prev_bbox from prev_frame
            prev_bbox = prev_frame[ymin:ymax, xmin:xmax]
            # Step 1: Extract new_bbox from current frame with the same coordinates
            xmin1 = max(0, xmin - 20)
            xmax1 = min(frame.shape[1], xmax + 20)
            ymin1 = max(0, ymin - 20)
            ymax1 = min(frame.shape[0], ymax + 20)
            new_bbox = frame[ymin1:ymax1, xmin1:xmax1]
            # Step 2: Calc match_template between previous and new bbox
            # Use padding
            matching = match_template(new_bbox, prev_bbox, pad_input=True)
            # Step 3: Then multiply matching by gauss function
            # Find argmax(matching * gauss)
            w = np.shape(matching)[0]
            h = np.shape(matching)[1]
            diff_y, diff_x = np.unravel_index(np.argmax(matching * gaussian(np.shape(matching), (h - 1) / 2, (w - 1) / 2, 1, 1)), np.shape(matching))
            diff_y -= w // 2
            diff_x -= h // 2
            detection = [label, xmin + diff_x, ymin + diff_y, min(frame.shape[1], xmax + diff_x), min(frame.shape[0], ymax + diff_y)]
            # Step 4: Append to detection list
            detections.append(detection)

        return detection_cast(detections)

    def update_frame(self, frame):
        if not self.frame_index:
            detections = self.init_tracklet(frame)
            self.save_detections(detections)
        elif self.frame_index % self.detection_rate == 0:
            detections = extract_detections(frame, labels=self.labels, min_confidence=0.4)
            detections = self.bind_tracklet(detections)
            self.save_detections(detections)
        else:
            detections = self.build_tracklet(frame)

        self.detection_history.append(detections)
        self.prev_frame = frame
        self.frame_index += 1

        if self.return_images:
            return draw_detections(frame, detections)
        else:
            return detections


def main():
    dirname = os.path.dirname(__file__)
    input_clip = VideoFileClip(os.path.join(dirname, 'data', 'test.mp4'))

    tracker = CorrelationTracker()
    input_clip.fl_image(tracker.update_frame).preview()


if __name__ == '__main__':
    main()

