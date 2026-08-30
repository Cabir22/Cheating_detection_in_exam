from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from ultralytics import YOLO

import pose_utils
from pose_utils import PoseConfig, DEFAULT_POSE_CONFIG
from tracker import TrackedBox


class GazeDirection(Enum):
    FORWARD = "forward"
    SIDE = "side"
    AWAY = "away"
    FACE_NOT_VISIBLE = "not_visible"


@dataclass
class HeadPoseResult:
    track_id: int
    yaw: float | None
    direction: GazeDirection


@dataclass
class GazeConfig:
    side_threshold: float = 20.0    # |yaw| bundan böyükdürsə -> SIDE
    away_threshold: float = 50.0    # |yaw| bundan böyükdürsə -> AWAY


DEFAULT_GAZE_CONFIG = GazeConfig()


# 1. Keypoint-lərdən yaw bucağını təxmin etmək

def estimate_yaw_from_keypoints(keypoints, pose_cfg: PoseConfig = DEFAULT_POSE_CONFIG):
    nose = keypoints[pose_utils.NOSE]
    left_shoulder = keypoints[pose_utils.LEFT_SHOULDER]
    right_shoulder = keypoints[pose_utils.RIGHT_SHOULDER]
    left_eye = keypoints[pose_utils.LEFT_EYE]
    right_eye = keypoints[pose_utils.RIGHT_EYE]

    if not pose_utils.is_visible(nose, pose_cfg):
        return None   # burun özü görünmürsə, etibarlı yaw hesablana bilməz

    if pose_utils.is_visible(left_shoulder, pose_cfg) and pose_utils.is_visible(right_shoulder, pose_cfg):
        center_x = (left_shoulder[0] + right_shoulder[0]) / 2
        ref_distance = abs(right_shoulder[0] - left_shoulder[0]) + 1e-6
    elif pose_utils.is_visible(left_eye, pose_cfg) and pose_utils.is_visible(right_eye, pose_cfg):
        center_x = (left_eye[0] + right_eye[0]) / 2
        ref_distance = abs(right_eye[0] - left_eye[0]) + 1e-6
    else:
        return None   # nə çiyin, nə göz -> yaw hesablamaq üçün məlumat kifayət deyil

    yaw_ratio = (nose[0] - center_x) / ref_distance
    return float(yaw_ratio * 90.0)



# 2. Bucağı kateqoriyaya çevirmək

def classify_direction(yaw, gaze_cfg: GazeConfig = DEFAULT_GAZE_CONFIG) -> GazeDirection:
    if yaw is None:
        return GazeDirection.FACE_NOT_VISIBLE

    abs_yaw = abs(yaw)
    if abs_yaw < gaze_cfg.side_threshold:
        return GazeDirection.FORWARD
    elif abs_yaw < gaze_cfg.away_threshold:
        return GazeDirection.SIDE
    else:
        return GazeDirection.AWAY



# 3. Hər şeyi birləşdirən əsas funksiya

def estimate_head_pose_yolo(pose_model: YOLO, frame, person: TrackedBox,
                             gaze_cfg: GazeConfig = DEFAULT_GAZE_CONFIG,
                             pose_cfg: PoseConfig = DEFAULT_POSE_CONFIG,
                             keypoints: np.ndarray | None = None) -> HeadPoseResult:

    if keypoints is None:
        keypoints = pose_utils.get_keypoints_for_person(pose_model, frame, person, pose_cfg)

    if keypoints is None:
        return HeadPoseResult(track_id=person.track_id, yaw=None,
                               direction=GazeDirection.FACE_NOT_VISIBLE)

    yaw = estimate_yaw_from_keypoints(keypoints, pose_cfg)
    direction = classify_direction(yaw, gaze_cfg)

    return HeadPoseResult(track_id=person.track_id, yaw=yaw, direction=direction)


# Vizual debug

_DIRECTION_COLORS = {
    GazeDirection.FORWARD: (0, 255, 0),
    GazeDirection.SIDE: (0, 255, 255),
    GazeDirection.AWAY: (0, 0, 255),
    GazeDirection.FACE_NOT_VISIBLE: (128, 128, 128),
}


def draw_head_pose_yolo(frame, person: TrackedBox, pose: HeadPoseResult):
    color = _DIRECTION_COLORS[pose.direction]
    label = f"     {pose.direction.value}"
    if pose.yaw is not None:
        label += f" ({pose.yaw:.0f} deg)"

    cv2.putText(frame, label, (person.x1, max(0, person.y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

