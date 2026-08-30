from dataclasses import dataclass

import numpy as np
import supervision as sv
from ultralytics import YOLO

from tracker import TrackedBox


@dataclass
class PhoneBox:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    owner_id: int | None = None


# Kadrda telefonları aşkarlayır
def detect_phones(model: YOLO, frame, conf: float = 0.65) -> list[PhoneBox]:

    # YOLO ilə detection və nəticəni supervision formatına çevirmək
    results = model.predict(frame, conf=conf, verbose=False)
    detections = sv.Detections.from_ultralytics(results[0])

    # Hər detection PhoneBox obyektinə çevir
    return [
        PhoneBox(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2), conf=float(c))
        for (x1, y1, x2, y2), c in zip(detections.xyxy.astype(int), detections.confidence)
    ]


def _box_center(x1, y1, x2, y2):
    return (x1 + x2) / 2, (y1 + y2) / 2


# Telefonu ən yaxın şəxsə uyğunlaşdırmaq
def associate_phones(phones: list[PhoneBox], persons: list[TrackedBox], max_distance = 300.0):
    for phone in phones:
        if not persons:
            continue
        pc = _box_center(phone.x1, phone.y1, phone.x2, phone.y2)
        best_id, best_dist = None, float("inf")
        # Ən yaxın şəxsi tap
        for p in persons:
            dist = ((pc[0] - (p.x1 + p.x2) / 2) ** 2 + (pc[1] - (p.y1 + p.y2) / 2) ** 2) ** 0.5
            if dist < best_dist:
                best_dist, best_id = dist, p.track_id
        phone.owner_id = best_id if best_dist <= max_distance else None
    return phones
