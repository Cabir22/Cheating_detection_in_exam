"""
head_pose_yolo.py üçün tam-pipeline test/demo skripti — real video açır,
cv2.imshow ilə göstərir. Model/video lazımdır, avtomatlaşdırılmış deyil.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "core"))

import cv2
from ultralytics import YOLO

from tracker import PersonTracker, draw_tracks
from head_pose_yolo import (
    estimate_head_pose_yolo,
    draw_head_pose_yolo,
    GazeConfig,
    DEFAULT_GAZE_CONFIG,
)
from pose_utils import PoseConfig, DEFAULT_POSE_CONFIG


def run_yolo_pose_test(video_path: str, person_model_path: str = r"E:\Exam  Cheating detection\models\yolo11m.pt",
                        pose_model_path: str = r"E:\Exam  Cheating detection\models\yolo11m-pose.pt",
                        tracker_cfg: str = "bytetrack.yaml",
                        display_scale: float = 0.6,
                        gaze_cfg: GazeConfig = DEFAULT_GAZE_CONFIG,
                        pose_cfg: PoseConfig = DEFAULT_POSE_CONFIG) -> None:

    person_tracker = PersonTracker(model_path=person_model_path, tracker_cfg=tracker_cfg)
    pose_model = YOLO(pose_model_path)

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

            for person in persons:
                pose = estimate_head_pose_yolo(pose_model, frame, person, gaze_cfg, pose_cfg)
                draw_head_pose_yolo(frame, person, pose)

            display = cv2.resize(frame, (0, 0), fx=display_scale, fy=display_scale)
            cv2.imshow("YOLO-Pose head direction test", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    #run_yolo_pose_test(video_path=r"E:\Exam  Cheating detection\Data\Group\v4\DSC_0110.MOV")
    run_yolo_pose_test(video_path=r"E:\Exam  Cheating detection\Data\Group\v4\DSC_0117.MOV")
