from dataclasses import dataclass
from itertools import combinations

import numpy as np

import pose_utils
from pose_utils import PoseConfig, DEFAULT_POSE_CONFIG


# Konfiqurasiya — yalnız əl-yaxınlığına aid dəyərlər

@dataclass
class HandProximityConfig:
    distance_threshold_ratio: float = 1.5    # əl-arası məsafə / orta çiyin eni bundan azdırsa "yaxın"
    required_consecutive_frames: int = 10    # bu qədər ardıcıl frame yaxın qalarsa "sustained" sayılır


DEFAULT_HAND_CONFIG = HandProximityConfig()


@dataclass
class PairProximityResult:
    track_id_1: int
    track_id_2: int
    distance: float | None   # normallaşdırılmış məsafə (çiyin eninə nisbətən); hesablana bilmirsə None
    is_sustained: bool       # N ardıcıl frame-dir yaxın qalıb 


# Köməkçilər

def _shoulder_width(keypoints, cfg: PoseConfig):
    left = keypoints[pose_utils.LEFT_SHOULDER]
    right = keypoints[pose_utils.RIGHT_SHOULDER]
    if not (pose_utils.is_visible(left, cfg) and pose_utils.is_visible(right, cfg)):
        return None
    return float(abs(right[0] - left[0]))


def _visible_wrists(keypoints, cfg: PoseConfig):
    wrists = []
    left = keypoints[pose_utils.LEFT_WRIST]
    right = keypoints[pose_utils.RIGHT_WRIST]
    if pose_utils.is_visible(left, cfg):
        wrists.append(("left", left))
    if pose_utils.is_visible(right, cfg):
        wrists.append(("right", right))
    return wrists


def _closest_hand_pair(kp_1, kp_2, cfg: PoseConfig):
    """İki nəfərin görünən əlləri arasında ən yaxın cütü (məsafə, əl_1, əl_2) qaytarır."""
    wrists_1 = _visible_wrists(kp_1, cfg)
    wrists_2 = _visible_wrists(kp_2, cfg)
    if not wrists_1 or not wrists_2:
        return None

    best = None
    for name_1, w_1 in wrists_1:
        for name_2, w_2 in wrists_2:
            dist = float(np.hypot(w_1[0] - w_2[0], w_1[1] - w_2[1]))  #[x, y, confidence]
            if best is None or dist < best[0]:
                best = (dist, name_1, name_2)
    return best



# Əsas sinif — ardıcıl frame-lər arasında state saxlayır

class HandProximityTracker:
    def __init__(self, hand_cfg = DEFAULT_HAND_CONFIG,
                 pose_cfg = DEFAULT_POSE_CONFIG):
        self.hand_cfg = hand_cfg
        self.pose_cfg = pose_cfg
        self._consecutive_counts = {}

    @staticmethod
    def _pair_key(id_1, id_2):
        return (id_1, id_2) if id_1 < id_2 else (id_2, id_1)

    def update(self, keypoints_by_id) -> list[PairProximityResult]:
        #Bir frame üçün bütün şəxs cütlərini yoxlayır və nəticələri qaytarır.
        results = []
        seen_keys = set() # bu frame-də görünən cütləri qeyd edirik

        # yalnız keypoint-i mövcud olan insanları götür
        valid_ids = [
            track_id 
            for track_id, kp in keypoints_by_id.items() 
            if kp is not None]

        for id_1, id_2 in combinations(valid_ids, 2):
            key = self._pair_key(id_1, id_2)
            seen_keys.add(key)

            kp_1 = keypoints_by_id[id_1]
            kp_2 = keypoints_by_id[id_2]

            width_1 = _shoulder_width(kp_1, self.pose_cfg)
            width_2 = _shoulder_width(kp_2, self.pose_cfg)
            closest = _closest_hand_pair(kp_1, kp_2, self.pose_cfg)

            if closest is None or width_1 is None or width_2 is None:
                # Lazımi keypoint-lər görünmür -> bu cüt üçün sayğacı sıfırla, siqnal vermə
                self._consecutive_counts[key] = 0
                results.append(PairProximityResult(
                    track_id_1=key[0], track_id_2=key[1],
                    distance=None, is_sustained=False,
                ))
                continue

            distance, _, _ = closest
            ref_width = (width_1 + width_2) / 2 + 1e-6
            normalized_distance = distance / ref_width

            is_close = normalized_distance < self.hand_cfg.distance_threshold_ratio
            if is_close:
                self._consecutive_counts[key] = self._consecutive_counts.get(key, 0) + 1
            else:
                self._consecutive_counts[key] = 0

            is_sustained = self._consecutive_counts[key] >= self.hand_cfg.required_consecutive_frames

            results.append(PairProximityResult(
                track_id_1=key[0], track_id_2=key[1],
                distance=normalized_distance, is_sustained=is_sustained,
            ))

        # Bu frame-də heç görünməyən cütlərin sayğacını təmizlə (yaddaş sızmasının qarşısını al)
        stale_keys = [k for k in self._consecutive_counts if k not in seen_keys]
        for k in stale_keys:
            del self._consecutive_counts[k]

        return results
