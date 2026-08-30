
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "core"))

import cv2
from ultralytics import YOLO

from tracker import PersonTracker
from detector import detect_phones, associate_phones


def run_detector_test(video_path: str,
                       person_model_path: str = r"E:\Exam  Cheating detection\models\yolo26m.pt",
                       phone_model_path: str = r"E:\Exam  Cheating detection\models\best11m.pt",
                       tracker_cfg: str = "bytetrack.yaml",
                       display_scale: float = 0.6):

     # Şəxsləri izləmək üçün tracker və telefon modeli yüklənir                   
    person_tracker = PersonTracker(model_path=person_model_path, tracker_cfg=tracker_cfg)
    phone_model = YOLO(phone_model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Video açıla bilmədi: {video_path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

             # Hər kadrdakı şəxslər izlənilir
            persons = person_tracker.update_frame(frame)
            # Telefonlar aşkarlanır və yaxınlıqdakı şəxslərlə əlaqələndirilir
            phones = associate_phones(detect_phones(phone_model, frame), persons)

            # Telefonlar qırmızı çərçivə ilə göstərilir əgər sahibinə bağlanıbsa owner_id göstərilir
            for phone in phones:
                cv2.rectangle(frame, (phone.x1, phone.y1), (phone.x2, phone.y2), (0, 0, 255), 2)
                label = f"phone id={phone.owner_id}" if phone.owner_id is not None else "phone"
                cv2.putText(frame, label, (phone.x1, max(0, phone.y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Şəxslər yaşıl çərçivə ilə göstərilir, tracking ID göstərilir
            for p in persons:
                cv2.rectangle(frame, (p.x1, p.y1), (p.x2, p.y2), (0, 255, 0), 2)
                cv2.putText(frame, f"id {p.track_id}", (p.x1, max(0, p.y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            display = cv2.resize(frame, (0, 0), fx=display_scale, fy=display_scale)
            cv2.imshow("frame", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Videonu yoxlamaq üçün
    # run_detector_test(video_path=r"E:\Exam  Cheating detection\Data\Group\v5\DSC_0080.MOV")
    run_detector_test(video_path=r"E:\Exam  Cheating detection\Data\Group\v5\DSC_0085.MOV")
    # run_detector_test(video_path=r"E:\Exam  Cheating detection\Data\Group\v4\DSC_0110.MOV")
    # run_detector_test(video_path=r"E:\Exam  Cheating detection\Data\Group\v5\DSC_0080.MOV")

