import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))

import cv2
import numpy as np

from hand_proximity import PairProximityResult
from tracker import TrackedBox


def draw_hand_proximity(frame, persons_by_id, results):
    """Yalnız 'sustained' (davamlı) əl yaxınlaşmalarını iki şəxs arasında xətt çəkərək göstərir."""
    for r in results:
        if not r.is_sustained:
            continue

        p1 = persons_by_id.get(r.track_id_1)
        p2 = persons_by_id.get(r.track_id_2)
        if p1 is None or p2 is None:
            continue

        c1 = (int((p1.x1 + p1.x2) / 2), int((p1.y1 + p1.y2) / 2))
        c2 = (int((p2.x1 + p2.x2) / 2), int((p2.y1 + p2.y2) / 2))

        cv2.line(frame, c1, c2, (0, 0, 255), 2)
        mid = ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2)
        cv2.putText(frame, f"hand-close ({r.distance:.2f})", mid,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return frame


# ---------------------------------------------------------------------------
# Xam siqnal-əsaslı alert label-lərini bounding box üzərində göstərmək
# ---------------------------------------------------------------------------

_PHONE_BG = (0, 0, 255)      # qırmızı (BGR)
_ALERT_BG = (0, 220, 220)    # sarı (BGR)
_TEXT_ON_PHONE_BG = (255, 255, 255)   # ağ — qırmızı fonda kontrast
_TEXT_ON_ALERT_BG = (0, 0, 0)         # qara — sarı fonda kontrast

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_FONT_THICKNESS = 1
_PAD = 4       # fon düzbucaqlısı ilə mətn arasında boşluq
_GAP = 3       # ard-arda label-lər arasında boşluq


def draw_alert_labels(frame, track_box, labels):
    y = track_box.y1 + _PAD

    for label in labels:
        bg_color = _PHONE_BG if "phone" in label.lower() else _ALERT_BG
        text_color = _TEXT_ON_PHONE_BG if bg_color is _PHONE_BG else _TEXT_ON_ALERT_BG

        (text_w, text_h), baseline = cv2.getTextSize(label, _FONT, _FONT_SCALE, _FONT_THICKNESS)
        box_h = text_h + baseline + 2 * _PAD

        top_left = (track_box.x1, y)
        bottom_right = (track_box.x1 + text_w + 2 * _PAD, y + box_h)
        cv2.rectangle(frame, top_left, bottom_right, bg_color, cv2.FILLED)

        text_origin = (track_box.x1 + _PAD, y + _PAD + text_h)
        cv2.putText(frame, label, text_origin, _FONT, _FONT_SCALE, text_color, _FONT_THICKNESS)

        y += box_h + _GAP

    return frame
