from ultralytics import YOLO
import cv2
import math

smoke_model = YOLO(r"D:\yolo_runs\smoke_finetuned_hardneg\weights\best.pt")
vehicle_model = YOLO("yolov8n.pt")

video_path = r"C:\Users\E028.26\Downloads\WhatsApp Video 2026-06-17 at 3.17.33 PM.mp4"

cap = cv2.VideoCapture(video_path)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    "smoke_vehicle_association_output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    smoke_results = smoke_model(frame, conf=0.15, imgsz=960, verbose=False)
    vehicle_results = vehicle_model(frame, conf=0.4, imgsz=640, verbose=False)

    smoke_boxes = []
    vehicle_boxes = []

    for box in smoke_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        smoke_boxes.append((x1, y1, x2, y2, conf))

    for box in vehicle_results[0].boxes:
        cls = int(box.cls[0])
        if cls in vehicle_classes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            name = vehicle_model.names[cls]
            vehicle_boxes.append((x1, y1, x2, y2, conf, name))

    suspected_index = None

    if smoke_boxes and vehicle_boxes:
        sx1, sy1, sx2, sy2, _ = smoke_boxes[0]
        smoke_center = center((sx1, sy1, sx2, sy2))

        min_dist = float("inf")

        for i, vehicle in enumerate(vehicle_boxes):
            vx1, vy1, vx2, vy2, _, _ = vehicle
            vehicle_center = center((vx1, vy1, vx2, vy2))

            dist = math.dist(smoke_center, vehicle_center)

            if dist < min_dist:
                min_dist = dist
                suspected_index = i

    for x1, y1, x2, y2, conf in smoke_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(frame, f"Smoke {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    for i, vehicle in enumerate(vehicle_boxes):
        x1, y1, x2, y2, conf, name = vehicle

        if i == suspected_index:
            color = (0, 255, 255)
            label = f"SUSPECTED {name} {conf:.2f}"
            thickness = 4
        else:
            color = (255, 0, 0)
            label = f"{name} {conf:.2f}"
            thickness = 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    out.write(frame)

cap.release()
out.release()

print("Done! Output saved as smoke_vehicle_association_output.mp4")