"""
reporting/visualizer.py üçün sintetik self-check — model/video lazım deyil.
Uydurma TrackedBox + label siyahısı ilə draw_alert_labels()-i çağırıb
nəticəni output/test_visualizer_labels.png-ə yazır (vizual yoxlama üçün).
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "core"))
sys.path.insert(0, str(_ROOT / "reporting"))

import cv2
import numpy as np

from tracker import TrackedBox
from visualizer import draw_alert_labels


def run_visualizer_self_check(output_path: str = None) -> str:
    frame = np.full((400, 500, 3), 40, dtype=np.uint8)

    box_1 = TrackedBox(track_id=1, x1=40, y1=40, x2=220, y2=320, conf=0.9)
    box_2 = TrackedBox(track_id=2, x1=280, y1=80, x2=460, y2=340, conf=0.9)

    for box in (box_1, box_2):
        cv2.rectangle(frame, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Id {box.track_id}", (box.x1, max(0, box.y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    draw_alert_labels(frame, box_1, ["PHONE", "LOOKING AROUND", "HAND RISK"])
    draw_alert_labels(frame, box_2, ["HAND RISK"])

    # Self-check: label fonları box-un daxilindədir, ID mətni ilə (y1 üstündə) toxunmur.
    assert 40 <= box_1.y1 + 4 <= box_1.y2, "label box-un daxilindən başlamalıdır"

    output_path = output_path or str(_ROOT / "output" / "test_visualizer_labels.png")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, frame)
    print(f"Self-check keçdi, nəticə yazıldı: {output_path}")
    return output_path


if __name__ == "__main__":
    run_visualizer_self_check()
