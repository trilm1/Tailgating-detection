"""ByteTrack adapter; update even on empty frames so track age is correct."""

from math import ceil
from inspect import signature
from types import SimpleNamespace

from .domain import Observation


class PersonTracker:
    def __init__(self, fps, config):
        from ultralytics.trackers.byte_tracker import BYTETracker

        # Detections between track_low_thresh and track_high_thresh only extend existing
        # tracks (ByteTrack's second association); new tracks need new_track_thresh.
        args = SimpleNamespace(track_high_thresh=config["track_high_thresh"], track_low_thresh=0.1,
                               new_track_thresh=config["new_track_thresh"], match_thresh=0.8,
                               fuse_score=True, track_buffer=ceil(30 * config["tracker_buffer_s"]))
        if "frame_rate" in signature(BYTETracker).parameters:
            self.tracker = BYTETracker(args, frame_rate=fps)
        else:
            # New Ultralytics accepts the buffer directly in frames.
            args.track_buffer = max(1, int(fps / 30 * args.track_buffer))
            self.tracker = BYTETracker(args)

    def update(self, detections, t_s, frame):
        tracks = self.tracker.update(detections, frame)
        return [Observation(int(row[4]), t_s, tuple(float(v) for v in row[:4]))
                for row in tracks]
