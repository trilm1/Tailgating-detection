# Tailgating Detection

## 1. Design

The system takes a video, access-grant timestamps, and a door-threshold configuration.
YOLO11m detects people in each frame, ByteTrack maintains their IDs, and the crossing logic
checks whether each person crosses the threshold in the entry direction. For applications
where performance is a priority and the indoor camera view is relatively stable, an option
would be to use OpenCV to detect substantial pixel changes and trigger the detector only
when motion occurs, reducing repeated inference during empty portions of the video.
Entries are counted per grant; two or more people trigger a `tailgating` alert. An entry
without a grant triggers an `unauthorised_entry` alert.

The detector uses pretrained YOLO weights without fine-tuning. During development,
YOLOv8n ran faster but frequently misclassified dogs near the door as people. YOLO11m reduced
this error at the cost of lower speed. Both achieved exact counts on 60/61 grants in the
initial five clips. I did not choose BoT-SORT with ReID because I wanted to keep the pipeline
simple and limit CPU processing costs. The dataset also has a limited number of participants,
so BoT-SORT's stronger ReID capability could adversely affect the scenarios I staged.
The model, weights source, and license are listed in the [README](README.md#models-and-libraries).

The detector confidence threshold is 0.1; the high-score association and new-track thresholds
are 0.25. ByteTrack still uses low-confidence detections to associate existing tracks when
people are occluded. Lost tracks are retained for approximately one second. An additional
ID-stitching step handles a new ID appearing near a recently lost track, or a new ID that
substantially overlaps the old track in its final frame. A track that has already emitted
a crossing cannot be used as a stitching target. This reduces some ID-switch errors but
can still incorrectly merge two people standing close together.

The door threshold is a directed line segment defined in a configuration file or through
`calibrate_door.py`. A crossing is confirmed only after a track has been observed beyond
the dead band on both sides and has crossed the door segment. The `point` mode uses the
bottom center of the bounding box; `footprint` requires its entire bottom edge to cross
the threshold. The crossing timestamp is interpolated between observations. No crossing
is inferred after an observation gap longer than one second. The same ID entering, exiting,
and entering again within one grant is counted only once.

## 2. Assumptions and Parameters

The door is assumed to be open between unlock and relock; the system does not detect the
physical state of the door. Grants in the dataset are simulated. Ground truth (GT) counts
unique person IDs within each grant, whereas the system counts track IDs. An ID switch
after an entry can therefore still cause duplicate counting.

The first five clips use `point` because the camera view cuts off people's feet at the
image boundary. The threshold is (350.480)→(400.280), with a 6 px margin and
`min_box_height_frac = 0.25` to discard boxes containing only the top of a person's head
near the camera. The four additional clips use `footprint`, have separate configurations,
and do not enable this height filter.

| Parameter | Value and Rationale |
|---|---|
| Dead band | max(6 px, 0.03 × bounding-box height), reducing false crossings caused by box jitter |
| Grace after relock | 0.5 s to allow for door-closing delay; testing 0/0.5/1 s with YOLOv8n produced the same results on the initial group |
| Lead before unlock | 0.1 s to account for annotation/interpolation error; one crossing occurred 0.02 s before unlock |
| ID stitching | Maximum distance of 0.35 × bounding-box height or IoU ≥ 0.5, within approximately one second |

Parameters were selected using the initial five clips. When replaying YOLO11m detections,
disabling the lead allowance reduced the number of correctly counted grants from 60 to 59;
disabling ID stitching retained 60/61. Height-filter values of 0.22/0.25/0.28 yielded
60/60/59 correctly counted grants, respectively.

## 3. Results

The dataset contains nine clips. The table below uses the final run in `results/final` and covers
73 grants; the walk-past clip without grants is evaluated separately. Precision, recall,
and F1 are calculated for alerts per grant. Count accuracy is the proportion of grants
whose predicted entry count exactly matches ground truth.

| Group | Grants | TP/FP/FN/TN | Precision | Recall | F1 | Exact Count |
|---|---:|---|---:|---:|---:|---:|
| Initial 5 clips | 61 | 12/0/0/49 | 1.000 | 1.000 | 1.000 | 60/61 |
| Additional 4 clips | 12 | 6/0/1/5 | 1.000 | 0.857 | 0.923 | 10/12 |
| Total, final run | 73 | 18/0/1/54 | 1.000 | 0.947 | 0.973 | 70/73 (95.89%) |

[Per-clip metrics](results/final/metrics_table.md) and
[confusion by scenario](results/final/confusion_by_scenario.md).

Clip …9613 covers scenario 6: people walk past without crossing the threshold, and there
are no grants. A header-only GT file is correct; the system produces `events.json = []`
with no alerts, matching the expected result. This clip is excluded from grant-level count
accuracy and confusion statistics.

The full YOLO11m pipeline achieved approximately 3.77 FPS for the initial group and 3.73 FPS
for the additional group, with an overall processing speed of 3.76 FPS. All nine clips were
processed on an i7-8750H/Windows laptop using CPU inference with four threads.
FPS includes video decoding, detection, tracking,
counting, rendering, and video writing; it excludes model loading and post-export MP4 validation.

All 47 tests pass, covering crossing logic, grant association, ID retention through brief
detection gaps, and output validation. Synthetic trajectory tests do not replace detector
evaluation on real video. All nine final annotated videos decode completely. Good performance
on the initial group does not generalize to every video, as the failures below demonstrate.

## 4. Failures and Limitations

| Case | Observation, Cause, and Potential Improvement |
|---|---|
| …5906/g01, 6–8 s | A person carrying a large basin is assigned two track IDs, 5 and 6, causing a false tailgating alert. The basin obscures the body and destabilizes detection/tracking. Duplicate boxes could be checked across multiple frames; merging solely by IoU risks removing a real second person. |
| …6909/g04, 68–69 s | Two people enter, but the count is 0. Bounding boxes and IDs are still present; the `footprint` condition may fail to establish observations on both sides when boxes are near the image boundary. The footage also cuts off partway through the crossing because of a recording issue. |

Failure images are available in [results/failures](results/failures). For …7261/g14, GT records
an entry followed by an exit, but the system does not count the entry. The camera angle does not
show the feet, making the annotation difficult to verify.

Overlapping bounding boxes remain a common issue that the current pretrained model does not resolve.

The system depends on a view of both sides of the threshold, detection quality, and stable
IDs. Complete occlusion, a person first appearing directly at the doorway, or an ID switch
after entry can cause missed entries, overcounting, or lost tracks. Overlapping grants can
also lead to incorrect association. These limitations are retained for evaluation; the work
does not include task-specific training or a complete solution to occlusion.

## 5. Deployment Across 500 Doors

Each door needs its own configuration and a verified threshold image. Actual door-state
signals should be used instead of inferring the state from relock timestamps. Separate
calibration data is needed for different camera angles and lighting conditions.

Real-time performance could be optimized through C++, with DeepStream, TensorRT, and
system-level performance optimizations as proposed options. The model would need fine-tuning
to detect people more accurately under different conditions, together with optimized trackers
using ReID for better performance in real deployments. Additional rules or models trained
specifically for difficult cases such as overlapping people or object occlusion could help
address two people being detected as a single bounding box.

Batch processing and GPU dataflow scheduling could improve throughput. Where possible,
processing nodes could serve individual areas or groups of doors instead of transmitting
all video to a central server.

## 6. Use of AI Coding Assistants

I used Codex and Claude Code to help implement my design, write tests, check logic, and edit
documentation. I handled and reviewed the design, video recording, annotation, and final
decisions. I am confident in my understanding of most of the code and the repository's
processing flow.

## 7. Time Spent

| Task | Actual Hours |
|---|---:|
| Requirements analysis and design | 2 |
| Video recording and annotation | 3 |
| Implementation and testing | 5 |
| Evaluation and failure analysis | 3 |
| Report | 2 |
| Total | 15 |
