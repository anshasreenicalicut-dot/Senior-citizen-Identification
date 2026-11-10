
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
from openpyxl import Workbook, load_workbook

from src.face_detector import FaceAgeGenderDetector, FaceLocator


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "visits.csv"
EXCEL_PATH = OUTPUT_DIR / "visits.xlsx"

LOG_INTERVAL_SECONDS = 30
TRACK_MAX_DISTANCE = 80
TRACK_MAX_MISSES = 10


def initialize_logs():
    headers = [
        "Date", "Time", "Age", "Gender",
        "Status", "Gender Confidence", "Track ID"
    ]

    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(headers)

    if not EXCEL_PATH.exists():
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Visits"
        sheet.append(headers)
        workbook.save(EXCEL_PATH)


def log_visit(prediction, track_id):
    now = datetime.now()

    row = [
        now.strftime("%Y-%m-%d"),
        now.strftime("%H:%M:%S"),
        prediction["age"],
        prediction["gender"],
        prediction["status"],
        prediction["gender_confidence"],
        track_id,
    ]

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(row)

    workbook = load_workbook(EXCEL_PATH)
    sheet = workbook["Visits"]
    sheet.append(row)
    workbook.save(EXCEL_PATH)


def match_tracks(faces, tracks, next_id, frame_number):
    """
    Match current face boxes to previous tracks using box centers.
    This is a lightweight approximation, not person re-identification.
    """
    detections = []

    for x, y, w, h in faces:
        center = (int(x + w / 2), int(y + h / 2))
        detections.append({
            "box": (int(x), int(y), int(w), int(h)),
            "center": center,
            "track_id": None,
        })

    used_track_ids = set()

    for detection in detections:
        cx, cy = detection["center"]
        best_id = None
        best_distance = TRACK_MAX_DISTANCE

        for track_id, track in tracks.items():
            if track_id in used_track_ids:
                continue

            tx, ty = track["center"]
            distance = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5

            if distance < best_distance:
                best_distance = distance
                best_id = track_id

        if best_id is None:
            best_id = next_id
            next_id += 1

        detection["track_id"] = best_id
        used_track_ids.add(best_id)

        tracks[best_id] = {
            "center": (cx, cy),
            "last_seen": frame_number,
        }

    # Remove tracks that have disappeared for too many frames.
    stale_ids = [
        track_id
        for track_id, track in tracks.items()
        if frame_number - track["last_seen"] > TRACK_MAX_MISSES
    ]

    for track_id in stale_ids:
        del tracks[track_id]

    return detections, next_id


def run(source):
    initialize_logs()

    locator = FaceLocator()
    predictor = FaceAgeGenderDetector()

    # Webcam: --source 0; video file: --source path/to/video.mp4
    video_source = int(source) if source.isdigit() else source
    capture = cv2.VideoCapture(video_source)

    if not capture.isOpened():
        raise RuntimeError(
            f"Cannot open video source: {source}. "
            "Check the webcam index or video path."
        )

    tracks = {}
    next_id = 1
    frame_number = 0
    last_logged = {}

    print("Video started. Press Q to quit.")
    print(f"CSV log: {CSV_PATH}")
    print(f"Excel log: {EXCEL_PATH}")

    try:
        while True:
            ok, frame = capture.read()

            if not ok:
                break

            frame_number += 1
            faces = locator.detect(frame)

            detections, next_id = match_tracks(
                faces, tracks, next_id, frame_number
            )

            for detection in detections:
                x, y, w, h = detection["box"]
                track_id = detection["track_id"]

                face_crop = frame[y:y + h, x:x + w]
                prediction = predictor.predict(face_crop)

                if prediction is None:
                    continue

                color = (
                    (0, 0, 255)
                    if prediction["senior"]
                    else (255, 0, 0)
                )

                label = (
                    f'ID {track_id} | Age ~{prediction["age"]} | '
                    f'{prediction["gender"]}'
                )

                status = prediction["status"]

                cv2.rectangle(
                    frame, (x, y), (x + w, y + h), color, 2
                )

                cv2.putText(
                    frame,
                    label,
                    (x, max(20, y - 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

                cv2.putText(
                    frame,
                    status,
                    (x, max(40, y - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

                now = time.time()
                if now - last_logged.get(track_id, 0) >= (
                    LOG_INTERVAL_SECONDS
                ):
                    log_visit(prediction, track_id)
                    last_logged[track_id] = now

            cv2.putText(
                frame,
                f"Faces detected: {len(detections)}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Senior Citizen Identification", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        capture.release()
        cv2.destroyAllWindows()

    print("Video stopped.")
    print(f"Records saved in {CSV_PATH} and {EXCEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index (0) or video filename",
    )
    args = parser.parse_args()

    run(args.source)
