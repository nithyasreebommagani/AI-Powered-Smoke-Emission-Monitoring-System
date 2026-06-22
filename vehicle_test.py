from ultralytics import YOLO

# Vehicle detection model
model = YOLO("yolov8n.pt")

# Video path
video_path = r"C:\Users\E028.26\Downloads\15035735_2160_3840_30fps.mp4"
# Run detection
model.predict(
    source=video_path,
    conf=0.4,
    save=True
)

print("Vehicle detection completed!")