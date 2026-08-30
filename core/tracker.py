from dataclasses import dataclass

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO



# Bir şəxsin izləmə (tracking) məlumatını saxlayır
@dataclass
class TrackedBox:
    track_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float


# YOLO + ByteTrack ilə insanları izləyən class
class PersonTracker:
    def __init__(self, model_path, device="cuda", tracker_cfg="bytetrack.yaml",
             conf=0.3, iou=0.4, imgsz=1280):
        self.model = YOLO(model_path)
        self.model.fuse()
        self.model.to(device)

        self.tracker_cfg = tracker_cfg
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

    def update_frame(self, frame) -> list[TrackedBox]:
        results = self.model.track(
            frame,
            classes=[0],            # yalnız insan
            iou=self.iou,
            conf=self.conf,
            persist=True,
            imgsz=self.imgsz,
            verbose=False,
            tracker=self.tracker_cfg,
        )

        return extract_tracks(results)



# YOLO nəticəsini TrackedBox siyahısına çevirir
def extract_tracks(results) -> list[TrackedBox]:
    detections = sv.Detections.from_ultralytics(results[0])

    # Track tapılmayıbsa boş siyahı qaytarır
    if detections.tracker_id is None:
        return []

    return [
        TrackedBox(int(tid), int(x1), int(y1), int(x2), int(y2), float(conf))
        for (x1, y1, x2, y2), tid, conf in zip(
            detections.xyxy.astype(int), detections.tracker_id, detections.confidence)
    ]



# Vizual debug — annotasiya çəkmək

# Track ID-yə uyğun rənglə bbox çəkən annotator
_BOX_ANNOTATOR = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK, thickness=2)


# Frame üzərinə bbox və ID annotasiyası çəkir
def draw_tracks(frame, tracks: list[TrackedBox]):
    if tracks:
        detections = sv.Detections(
            xyxy=np.array([[t.x1, t.y1, t.x2, t.y2] for t in tracks], dtype=float),
            tracker_id=np.array([t.track_id for t in tracks], dtype=int),
        )
        _BOX_ANNOTATOR.annotate(scene=frame, detections=detections)

   # Hər şəxs üçün ID mətnini göstər
    for t in tracks:
        cv2.putText(
            frame, f"Id {t.track_id}", (t.x1, max(0, t.y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2,
        )
    return frame