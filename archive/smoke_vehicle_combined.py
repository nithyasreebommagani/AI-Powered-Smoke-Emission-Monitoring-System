from ultralytics import YOLO
import cv2

smoke_model = YOLO(r"D:\yolo_runs\smoke_finetuned_hardneg\weights\best.pt")
vehicle_model = YOLO("yolov8n.pt")

video_path = r"C:\Users\E028.26\Downloads\WhatsApp Video 2026-06-17 at 3.17.33 PM.mp4"
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    "smoke_vehicle_output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

while True:
    ret, frame = cap.read()
    if not ret:
        break

    smoke_results = smoke_model(frame, conf=0.15, imgsz=960, verbose=False)
    vehicle_results = vehicle_model(frame, conf=0.4, verbose=False)

    # Draw smoke boxes
    for box in smoke_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(frame, f"Smoke {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Draw vehicle boxes
    for box in vehicle_results[0].boxes:
        cls = int(box.cls[0])
        if cls in vehicle_classes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            name = vehicle_model.names[cls]
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    out.write(frame)

cap.release()
out.release()

print("Done! Output saved as smoke_vehicle_output.mp4")