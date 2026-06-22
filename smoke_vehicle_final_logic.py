from ultralytics import YOLO
import cv2
import math
from collections import defaultdict, deque

# Models
smoke_model = YOLO(r"D:\yolo_runs\smoke_finetuned_hardneg\weights\best.pt")
vehicle_model = YOLO("yolov8n.pt")

# Video path
video_path = r"C:\Users\E028.26\Downloads\fixed_video.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 30

out = cv2.VideoWriter(
    "final_suspect_output_improved.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# 150 frames = 5 seconds at 30 FPS
vehicle_smoke_history = defaultdict(lambda: deque(maxlen=150))

last_vehicle_boxes = {}
last_seen_frame = {}
id_alias = {}

frame_no = 0
def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def canonical_id(vehicle_id):
    visited = set()

    while vehicle_id in id_alias:
        if vehicle_id in visited:
            break

        visited.add(vehicle_id)
        vehicle_id = id_alias[vehicle_id]

    return vehicle_id


def merge_id_if_needed(vehicle_id, box):
    new_c = center(box)

    for old_id, old_box in last_vehicle_boxes.items():
        old_c = center(old_box)

        # Keep lost vehicle memory alive for 5 seconds
        if frame_no - last_seen_frame.get(old_id, 9999) <= 150:
            if dist(new_c, old_c) < 250:
                if vehicle_id != old_id:
                    id_alias[vehicle_id] = old_id
                return old_id

    return vehicle_id


def smoke_near_vehicle(smoke_box, vehicle_box):
    sx1, sy1, sx2, sy2 = smoke_box
    vx1, vy1, vx2, vy2 = vehicle_box

    vw = vx2 - vx1
    vh = vy2 - vy1

    expanded_vehicle = (
        vx1 - int(1.5 * vw),
        vy1 - int(1.5 * vh),
        vx2 + int(2.5 * vw),
        vy2 + int(2.0 * vh)
    )

    ex1, ey1, ex2, ey2 = expanded_vehicle
    smoke_c = center(smoke_box)

    return ex1 <= smoke_c[0] <= ex2 and ey1 <= smoke_c[1] <= ey2
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_no += 1

    if frame_no % 30 == 0:
        print("Processing frame:", frame_no)

    smoke_results = smoke_model(frame, conf=0.18, imgsz=960, verbose=False)

    vehicle_results = vehicle_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=vehicle_classes,
        conf=0.25,
        iou=0.5,
        verbose=False
    )

    smoke_boxes = []

    for box in smoke_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        smoke_boxes.append((x1, y1, x2, y2, conf))

    vehicle_data = []

    if vehicle_results[0].boxes.id is not None:
        for box in vehicle_results[0].boxes:
            cls = int(box.cls[0])
            raw_id = int(box.id[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            name = vehicle_model.names[cls]

            current_box = (x1, y1, x2, y2)

            vehicle_id = merge_id_if_needed(raw_id, current_box)
            vehicle_id = canonical_id(vehicle_id)

            last_vehicle_boxes[vehicle_id] = current_box
            last_seen_frame[vehicle_id] = frame_no

            vehicle_data.append((vehicle_id, x1, y1, x2, y2, conf, name))

    for vehicle in vehicle_data:
        vehicle_id, vx1, vy1, vx2, vy2, vconf, name = vehicle
        vehicle_box = (vx1, vy1, vx2, vy2)

        smoke_found = False

        for smoke in smoke_boxes:
            sx1, sy1, sx2, sy2, sconf = smoke
            smoke_box = (sx1, sy1, sx2, sy2)

            if smoke_near_vehicle(smoke_box, vehicle_box):
                smoke_found = True
                break

        vehicle_smoke_history[vehicle_id].append(1 if smoke_found else 0)

    for sx1, sy1, sx2, sy2, sconf in smoke_boxes:
        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 0, 255), 3)
        cv2.putText(
            frame,
            f"Smoke {sconf:.2f}",
            (sx1, sy1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    for vehicle in vehicle_data:
        vehicle_id, x1, y1, x2, y2, conf, name = vehicle

        smoke_count = sum(vehicle_smoke_history[vehicle_id])
        checked = len(vehicle_smoke_history[vehicle_id])

        if checked >= 150 and smoke_count >= 60:
            color = (0, 0, 255)
            label = f"SUSPECT ID:{vehicle_id} {name} smoke:{smoke_count}/150"
            thickness = 4
        else:
            color = (255, 0, 0)
            label = f"ID:{vehicle_id} {name} smoke:{smoke_count}/150"
            thickness = 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    out.write(frame)

cap.release()
out.release()

print("Done! Output saved as final_suspect_output_improved.mp4")