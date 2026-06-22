from ultralytics import YOLO
import cv2
import math
import os
import csv
from collections import defaultdict, deque

# =====================================
# MODELS
# =====================================

smoke_model = YOLO(
    r"D:\yolo_runs\smoke_finetuned_hardneg\weights\best.pt"
)

vehicle_model = YOLO("yolov8n.pt")

plate_model = YOLO(
    r"C:\Users\E028.26\Downloads\best.pt"
)

# =====================================
# INPUT VIDEO
# =====================================

video_path = r"C:\Users\E028.26\Downloads\WhatsApp Video 2026-06-19 at 3.50.10 PM.mp4"

# =====================================
# OUTPUT FOLDERS
# =====================================

evidence_dir = "evidence"
os.makedirs(evidence_dir, exist_ok=True)

# =====================================
# CSV LOG
# =====================================

log_file = os.path.join(evidence_dir, "evidence_log.csv")

with open(log_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Vehicle_ID",
        "Vehicle_Type",
        "Timestamp",
        "Smoke_Count",
        "Frame_Number"
    ])

# =====================================
# TRACKER DATA
# =====================================

stable_tracks = {}
next_stable_id = 1
used_ids_this_frame = set()
track_age = defaultdict(int)
smoke_history = defaultdict(lambda: deque(maxlen=150))
saved_suspects = set()

# =====================================
# BEST FRAME STORAGE
# =====================================

best_vehicle_crop = {}
best_vehicle_frame = {}
best_vehicle_area = {}

# =====================================
# VEHICLE CLASSES
# =====================================

vehicle_classes = [
    2,  # car
    3,  # motorcycle
    5,  # bus
    7   # truck
]

# =====================================
# HELPER FUNCTIONS
# =====================================

def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = box_area((ix1, iy1, ix2, iy2))
    union = box_area(boxA) + box_area(boxB) - inter

    if union == 0:
        return 0

    return inter / union

# =====================================
# STABLE TRACKER
# =====================================

def match_or_create_stable_id(vehicle_box, vehicle_name, frame_no):
    global next_stable_id

    vc = center(vehicle_box)
    best_id = None
    best_score = -1

    for stable_id, data in stable_tracks.items():
        if stable_id in used_ids_this_frame:
            continue

        frames_missing = frame_no - data["last_seen"]
        if frames_missing > 120:
            continue

        old_box = data["box"]
        overlap = iou(vehicle_box, old_box)

        if overlap < 0.05:
            continue

        d = distance(vc, center(old_box))
        size_ratio = box_area(vehicle_box) / max(box_area(old_box), 1)

        if not (0.5 <= size_ratio <= 2.0):
            continue

        score = overlap * 5
        score += max(0, (150 - d) / 150)

        if vehicle_name == data["name"]:
            score += 1

        if score > best_score:
            best_score = score
            best_id = stable_id

    if best_id is not None:
        stable_tracks[best_id]["box"] = vehicle_box
        stable_tracks[best_id]["last_seen"] = frame_no
        stable_tracks[best_id]["name"] = vehicle_name
        used_ids_this_frame.add(best_id)
        return best_id

    stable_id = next_stable_id
    next_stable_id += 1

    stable_tracks[stable_id] = {
        "box": vehicle_box,
        "name": vehicle_name,
        "last_seen": frame_no
    }

    used_ids_this_frame.add(stable_id)
    return stable_id

# =====================================
# SMOKE ASSOCIATION
# =====================================

def smoke_near_vehicle(smoke_box, vehicle_box):
    sx1, sy1, sx2, sy2 = smoke_box
    vx1, vy1, vx2, vy2 = vehicle_box

    vw = vx2 - vx1
    vh = vy2 - vy1

    expanded = (
        vx1 - int(0.5 * vw),
        vy1 - int(0.5 * vh),
        vx2 + int(1.0 * vw),
        vy2 + int(1.0 * vh)
    )

    ex1, ey1, ex2, ey2 = expanded
    sc = center(smoke_box)

    return (ex1 <= sc[0] <= ex2 and ey1 <= sc[1] <= ey2)

# =====================================
# VIDEO INPUT
# =====================================

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()

print("Video opened successfully")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# =====================================
# OUTPUT VIDEO
# =====================================

out = cv2.VideoWriter(
    "stable_suspect_output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

frame_no = 0

# =====================================
# MAIN LOOP
# =====================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_no += 1

    if frame_no % 30 == 0:
        print("Processing frame:", frame_no)

    used_ids_this_frame.clear()

    # ----------------------
    # Smoke Detection
    # ----------------------

    smoke_results = smoke_model(
        frame,
        conf=0.18,
        imgsz=960,
        verbose=False
    )

    smoke_boxes = []

    if len(smoke_results) > 0:
        for box in smoke_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            smoke_boxes.append((x1, y1, x2, y2))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame, "Smoke", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )

    # ----------------------
    # Vehicle Detection
    # ----------------------

    vehicle_results = vehicle_model(
        frame,
        conf=0.20,
        imgsz=960,
        verbose=False
    )

    current_vehicles = []

    for box in vehicle_results[0].boxes:
        cls = int(box.cls[0])

        if cls not in vehicle_classes:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        vehicle_box = (x1, y1, x2, y2)
        name = vehicle_model.names[cls]

        stable_id = match_or_create_stable_id(vehicle_box, name, frame_no)
        track_age[stable_id] += 1
        current_vehicles.append((stable_id, vehicle_box, name))

    # ----------------------
    # PROCESS VEHICLES
    # ----------------------

    for stable_id, vehicle_box, name in current_vehicles:
        smoke_found = False

        for smoke_box in smoke_boxes:
            if smoke_near_vehicle(smoke_box, vehicle_box):
                smoke_found = True
                break

        smoke_history[stable_id].append(1 if smoke_found else 0)
        smoke_count = sum(smoke_history[stable_id])
        checked = len(smoke_history[stable_id])

        x1, y1, x2, y2 = vehicle_box
        crop = frame[y1:y2, x1:x2]
        area = (x2 - x1) * (y2 - y1)

        # ------------------
        # SAVE BEST FRAME
        # ------------------

        if crop.size > 0:
            if stable_id not in best_vehicle_area:
                best_vehicle_area[stable_id] = area
                best_vehicle_crop[stable_id] = crop.copy()
                best_vehicle_frame[stable_id] = frame.copy()
            elif area > best_vehicle_area[stable_id]:
                best_vehicle_area[stable_id] = area
                best_vehicle_crop[stable_id] = crop.copy()
                best_vehicle_frame[stable_id] = frame.copy()

        suspect = (
            checked >= 150
            and smoke_count >= 60
            and track_age[stable_id] >= 100
        )

        if suspect:
            color = (0, 0, 255)
            label = f"SUSPECT Vehicle-{stable_id} {name}"

            if stable_id not in saved_suspects:
                timestamp_sec = frame_no / fps
                minutes = int(timestamp_sec // 60)
                seconds = int(timestamp_sec % 60)
                time_string = f"{minutes:02d}:{seconds:02d}"

                frame_path = os.path.join(
                    evidence_dir, f"Vehicle_{stable_id}_frame.jpg"
                )
                crop_path = os.path.join(
                    evidence_dir, f"Vehicle_{stable_id}_crop.jpg"
                )

                if stable_id in best_vehicle_frame:
                    cv2.imwrite(frame_path, best_vehicle_frame[stable_id])

                if stable_id in best_vehicle_crop:
                    cv2.imwrite(crop_path, best_vehicle_crop[stable_id])

                    vehicle_crop = best_vehicle_crop[stable_id]
                    plate_results = plate_model(
                        vehicle_crop, conf=0.10, verbose=False
                    )

                    for plate_box in plate_results[0].boxes:
                        px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
                        plate_crop = vehicle_crop[py1:py2, px1:px2]

                        plate_path = os.path.join(
                            evidence_dir, f"Vehicle_{stable_id}_plate.jpg"
                        )
                        cv2.imwrite(plate_path, plate_crop)
                        print(f"Plate saved for Vehicle {stable_id}")

                with open(log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        stable_id, name, time_string, smoke_count, frame_no
                    ])

                print(f"Suspect {stable_id} saved")
                saved_suspects.add(stable_id)

        else:
            color = (255, 0, 0)
            label = f"Vehicle-{stable_id}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    out.write(frame)

# =====================================
# CLEANUP
# =====================================

cap.release()
out.release()

print()
print("Finished Processing")
print("Total suspects:", len(saved_suspects))
print("Output Video:", "stable_suspect_output.mp4")