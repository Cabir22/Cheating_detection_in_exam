import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "core"))
sys.path.insert(0, str(_ROOT / "reporting"))

import cv2
from ultralytics import YOLO

import pose_utils
from pose_utils import PoseConfig, DEFAULT_POSE_CONFIG
from tracker import PersonTracker, draw_tracks
from hand_proximity import HandProximityTracker, HandProximityConfig, DEFAULT_HAND_CONFIG
from visualizer import draw_hand_proximity


def run_hand_proximity_test(video_path: str,
                             person_model_path: str = r"E:\Exam  Cheating detection\models\yolo26m.pt",
                             pose_model_path: str = r"E:\Exam  Cheating detection\models\yolov8m-pose.pt",
                             tracker_cfg: str = "bytetrack.yaml",
                             display_scale: float = 0.6,
                             hand_cfg: HandProximityConfig = DEFAULT_HAND_CONFIG,
                             pose_cfg: PoseConfig = DEFAULT_POSE_CONFIG) -> None:
    person_tracker = PersonTracker(model_path=person_model_path, tracker_cfg=tracker_cfg)
    pose_model = YOLO(pose_model_path)
    hand_tracker = HandProximityTracker(hand_cfg, pose_cfg)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Video açıla bilmədi: {video_path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            persons = person_tracker.update_frame(frame)
            draw_tracks(frame, persons)

            keypoints_by_id = pose_utils.get_keypoints_for_all(pose_model, frame, persons, pose_cfg)
            results = hand_tracker.update(keypoints_by_id)

            persons_by_id = {p.track_id: p for p in persons}
            draw_hand_proximity(frame, persons_by_id, results)

            display = cv2.resize(frame, (0, 0), fx=display_scale, fy=display_scale)
            cv2.imshow("Hand proximity test", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Öz video yolunla əvəz et
    run_hand_proximity_test(video_path=r"E:\Exam  Cheating detection\Data\Group\v4\DSC_0110.MOV")
