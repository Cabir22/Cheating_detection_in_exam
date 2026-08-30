import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "core"))

import cv2

from tracker import PersonTracker, draw_tracks


def run_tracker_on_video(
    tracker: PersonTracker,
    input_video_path,
    show_video = True,
    save_video = False,
    output_video_path = "output_video.mp4",
    display_scale = 0.6,
    frame_skip = 1,
):

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError(f"Video açıla bilmədi: {input_video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Müəyyən frame-lərdə tracking 
            if frame_count % frame_skip == 0:
                tracks = tracker.update_frame(frame)
                draw_tracks(frame, tracks)

            if save_video and out is not None:
                out.write(frame)

            if show_video:
                display_frame = cv2.resize(frame, (0, 0), fx=display_scale, fy=display_scale)
                cv2.imshow("frame", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
    finally:
        cap.release()
        if out is not None:
            out.release()
        if show_video:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    VIDEO_PATH = r"E:\Exam  Cheating detection\Data\Group\v5\DSC_0080.MOV"

    tracker = PersonTracker(
        model_path=r"E:\Exam  Cheating detection\models\yolo26m.pt",
        device="cuda",
        tracker_cfg="bytetrack.yaml",
        conf=0.3,
        iou=0.4,
        imgsz=960,
    )

    run_tracker_on_video(
        tracker,
        input_video_path=VIDEO_PATH,
        show_video=True,
        save_video=False,
        frame_skip=1,   # sürət problemi olsa 2-yə dəyişirik
    )
