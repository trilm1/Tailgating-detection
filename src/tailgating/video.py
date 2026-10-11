"""Validate an encoded video by decoding every frame before accepting it."""

import cv2


def validate_video(path, expected_frames=None, expected_size=None):
    capture = cv2.VideoCapture(str(path))
    count = 0
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot decode output video: {path}")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if expected_size is not None and (frame.shape[1], frame.shape[0]) != tuple(expected_size):
                raise ValueError(f"Output video dimensions changed: {path}")
            count += 1
        if not count or (expected_frames is not None and count != expected_frames):
            raise ValueError(f"Incomplete output video: {count}/{expected_frames} frames: {path}")
        return count
    finally:
        capture.release()
