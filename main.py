import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _sub in ("core", "reporting"):
    sys.path.insert(0, str(_ROOT / _sub))

import cv2
import supervision as sv
from ultralytics import YOLO

import pose_utils
from tracker import PersonTracker, draw_tracks
from head_pose_yolo import estimate_head_pose_yolo, GazeDirection, HeadPoseResult
from hand_proximity import HandProximityTracker, HandProximityConfig, PairProximityResult
from detector import PhoneBox, detect_phones, associate_phones
from visualizer import draw_alert_labels

# Əl-yaxınlığı üçün asan-tənzimlənən sabitlər.
HAND_DISTANCE_RATIO = 1.5
HAND_CONSECUTIVE_FRAMES = 10


def _should_flag_hand_risk(pair_result: PairProximityResult,
                            direction_1: GazeDirection, direction_2: GazeDirection):
    if not pair_result.is_sustained:
        return False
    anomaly_directions = (GazeDirection.SIDE, GazeDirection.AWAY)
    return direction_1 in anomaly_directions or direction_2 in anomaly_directions


def _build_labels(track_id: int, direction: GazeDirection,
                   phones: list[PhoneBox], hand_results: list[PairProximityResult],
                   pose_by_id: dict[int, HeadPoseResult]):
    labels = []

    if any(p.owner_id == track_id for p in phones):
        labels.append("PHONE")

    if direction in (GazeDirection.SIDE, GazeDirection.AWAY):
        labels.append("LOOKING AROUND")

    for r in hand_results:
        if track_id not in (r.track_id_1, r.track_id_2):
            continue
        dir_1 = pose_by_id[r.track_id_1].direction
        dir_2 = pose_by_id[r.track_id_2].direction
        if _should_flag_hand_risk(r, dir_1, dir_2):
            labels.append("HAND RISK")
            break

    return labels


def run_pipeline(video_path,
                  person_model_path = r"E:\Exam  Cheating detection\models\yolo26m.pt",
                  pose_model_path = r"E:\Exam  Cheating detection\models\yolov8m-pose.pt",
                  phone_model_path = r"E:\Exam  Cheating detection\models\best11m.pt",
                  tracker_cfg = "bytetrack.yaml",
                  output_dir = "output",
                  show_live = False,
                  max_frames = None):

    person_tracker = PersonTracker(model_path=person_model_path, tracker_cfg=tracker_cfg)
    pose_model = YOLO(pose_model_path).to("cuda")
    phone_model = YOLO(phone_model_path).to("cuda")
    hand_tracker = HandProximityTracker(HandProximityConfig(
        distance_threshold_ratio=HAND_DISTANCE_RATIO,
        required_consecutive_frames=HAND_CONSECUTIVE_FRAMES,
    ))

    video_info = sv.VideoInfo.from_video_path(video_path)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / f"{Path(video_path).stem}_annotated.mp4"


    with sv.VideoSink(str(out_path), video_info) as sink:
        for frame in sv.get_video_frames_generator(video_path, end=max_frames):
            persons = person_tracker.update_frame(frame)
            keypoints_by_id = pose_utils.get_keypoints_for_all(pose_model, frame, persons)
            hand_results = hand_tracker.update(keypoints_by_id)
            phones = associate_phones(detect_phones(phone_model, frame), persons)

            pose_by_id = {
                person.track_id: estimate_head_pose_yolo(
                    pose_model, frame, person, keypoints=keypoints_by_id.get(person.track_id))
                for person in persons
            }

            draw_tracks(frame, persons)
            for person in persons:
                direction = pose_by_id[person.track_id].direction
                labels = _build_labels(person.track_id, direction, phones, hand_results, pose_by_id)
                if labels:
                    draw_alert_labels(frame, person, labels)

            sink.write_frame(frame)

            if show_live:
                cv2.imshow("Proctoring pipeline", cv2.resize(frame, (0, 0), fx=0.6, fy=0.6))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    if show_live:
        cv2.destroyAllWindows()

    print(f"Annotasiya olunmuş video yazıldı: {out_path}")
    return str(out_path)

