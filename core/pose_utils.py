from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from tracker import TrackedBox

# YOLO-Pose keypoint indeksləri
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10



# Konfiqurasiya — crop/predict/confidence ilə bağlı hər şey

@dataclass
class PoseConfig:
    keypoint_conf_threshold = 0.3   # bu əminlikdən aşağı nöqtələr "görünmür" sayılır
    pose_predict_conf = 0.3         # YOLO-Pose modelinin predict conf-u
    crop_padding = 0.1              # person bbox-u ətrafına əlavə olunan padding


DEFAULT_POSE_CONFIG = PoseConfig()

#keypoint-in görünüb-görünmədiyini yoxlayır.
def is_visible(keypoint, cfg: PoseConfig = DEFAULT_POSE_CONFIG):  
    return keypoint[2] > cfg.keypoint_conf_threshold  # keypoint = [x, y, confidence]



#frame-dən yalnız konkret şəxsin olduğu hissəni kəsir.
def crop_person_for_pose(frame, person: TrackedBox, cfg: PoseConfig = DEFAULT_POSE_CONFIG):
    h, w = frame.shape[:2]
    box_w = person.x2 - person.x1
    box_h = person.y2 - person.y1

    x1 = max(0, int(person.x1 - box_w * cfg.crop_padding))
    y1 = max(0, int(person.y1 - box_h * cfg.crop_padding))
    x2 = min(w, int(person.x2 + box_w * cfg.crop_padding))
    y2 = min(h, int(person.y2 + box_h * cfg.crop_padding))

    return frame[y1:y2, x1:x2], x1, y1


# Crop üzərində YOLO-Pose ilə keypoint-ləri hesablayır
def get_pose_keypoints(pose_model: YOLO, person_crop, cfg: PoseConfig = DEFAULT_POSE_CONFIG):
    if person_crop.size == 0:
        return None

    results = pose_model.predict(person_crop, verbose=False, conf=cfg.pose_predict_conf)

    if len(results[0].boxes) == 0 or results[0].keypoints is None:
        return None

    return results[0].keypoints.data[0].cpu().numpy()



# Bir şəxs üçün frame koordinatlarında keypoint-ləri qaytarır
def get_keypoints_for_person(pose_model, frame, person: TrackedBox, cfg: PoseConfig = DEFAULT_POSE_CONFIG):
    crop, offset_x, offset_y = crop_person_for_pose(frame, person, cfg)
    keypoints = get_pose_keypoints(pose_model, crop, cfg)

    if keypoints is None:
        return None

    keypoints = keypoints.copy()
    keypoints[:, 0] += offset_x
    keypoints[:, 1] += offset_y
    return keypoints   # indi tam frame koordinatlarındadır


# Frame-dəki bütün şəxslər üçün keypoint-ləri hesablayır
def get_keypoints_for_all(pose_model: YOLO, frame, persons: list[TrackedBox],
                           cfg: PoseConfig = DEFAULT_POSE_CONFIG):

    return {person.track_id: get_keypoints_for_person(pose_model, frame, person, cfg)
            for person in persons}