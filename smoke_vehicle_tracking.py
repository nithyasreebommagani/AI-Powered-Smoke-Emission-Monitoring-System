from ultralytics import YOLO
import cv2

# Vehicle detector
vehicle_model = YOLO("yolov8n.pt")

video_path = r"C:\Users\E028.26\Downloads\WhatsApp Video 2026-06-17 at 3.17.33 PM.mp4"

cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = vehicle_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[2, 3, 5, 7],
        conf=0.25,
        iou=0.5,
        verbose=False
    )

    annotated = results[0].plot()

    cv2.imshow("Vehicle Tracking", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()