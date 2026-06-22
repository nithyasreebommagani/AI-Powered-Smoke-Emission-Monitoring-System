from ultralytics import YOLO
import cv2

plate_model = YOLO(
    r"C:\Users\E028.26\Downloads\best.pt"
)

img = cv2.imread(
    r"D:\combined_smoke_dataset\evidence\Vehicle_4_frame.jpg"
)

results = plate_model(img, conf=0.10)

for box in results[0].boxes:

    x1, y1, x2, y2 = map(
        int,
        box.xyxy[0]
    )

    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

cv2.imwrite(
    "plate_test_result.jpg",
    img
)

print("Saved: plate_test_result.jpg")